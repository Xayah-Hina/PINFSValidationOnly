import torch
import torchvision
import numpy as np
import argparse
import abc
import tqdm


class SingleVarianceNetwork(torch.nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', torch.nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


class InputEncoder(torch.nn.Module):
    def __init__(self, input_dims: int):
        super().__init__()
        self.output_dims = input_dims

    def forward(self, inputs: torch.Tensor):
        return inputs


class SinusoidalEncoder(InputEncoder):
    def __init__(self, input_dims: int, num_freq: int, include_input: bool = True):
        super().__init__(input_dims)
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_freq - 1, steps=num_freq)
        self.output_dims = 2 * num_freq * input_dims
        if include_input:
            self.output_dims += input_dims

    def forward(self, inputs: torch.Tensor):
        outputs = []
        if self.include_input:
            outputs.append(inputs)
        for freq in self.freq_bands:
            mult = inputs * freq
            outputs.append(torch.sin(mult))
            outputs.append(torch.cos(mult))
        return torch.cat(outputs, -1)


def positional_encoder(multires: int, input_dims: int = 3) -> tuple[InputEncoder, int]:
    encoder = SinusoidalEncoder(input_dims, multires) if multires > 0 and input_dims > 0 else InputEncoder(input_dims)
    return encoder, encoder.output_dims


class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        with torch.no_grad():
            if is_first:
                width = 1 / in_features
            else:
                width = np.sqrt(6 / in_features) / self.omega_0
            self.linear.weight.uniform_(-width, width)

    def forward(self, inputs):
        return torch.sin(self.omega_0 * self.linear(inputs))


# Velocity Model
class SIREN_vel(torch.nn.Module):
    def __init__(self, D=6, W=128, input_ch=4, skips=(), first_omega_0=30.0, unique_first=False, fading_fin_step=0):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_vel, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0

        hidden_omega_0 = 1.0

        self.hid_linears = torch.nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0, is_first=unique_first)] +
            [SineLayer(W, W, omega_0=hidden_omega_0)
             if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D - 1)]
        )

        self.vel_linear = torch.nn.Linear(W, 3)

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - vel_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def forward(self, x):
        h = x
        h_layers = []
        for i, l in enumerate(self.hid_linears):
            h = self.hid_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([x, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        vel_out = self.vel_linear(h)

        return vel_out


class RadianceField(torch.nn.Module):
    def __init__(self, output_sdf: bool = False):
        """
        Args:
            output_sdf: indicate that the returned extra part of forward() contains sdf
        """
        super().__init__()
        self.output_sdf = output_sdf

    @abc.abstractmethod
    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None) \
            -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """
        Args:
            x: input points [shape, 3 or 4]
            dirs: input directions [shape, 3]
            cond: extra conditions
        Returns:
            rgb [shape, 3], sigma [shape, 1] if applicable, extra outputs as dict
        """
        pass

    # pinf fading support, optional

    def update_fading_step(self, fading_step: int):
        pass

    def print_fading(self):
        pass


class SIREN_NeRFt(RadianceField):
    def __init__(self, D=8, W=256, input_ch=4, skips=(4,), use_viewdirs=False, first_omega_0=30.0, unique_first=False,
                 fading_fin_step=0, **kwargs):
        """
        fading_fin_step: >0, to fade in layers one by one, fully faded in when self.fading_step >= fading_fin_step
        """

        super(SIREN_NeRFt, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = 3 if use_viewdirs else 0
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.fading_step = 0
        self.fading_fin_step = fading_fin_step if fading_fin_step > 0 else 0

        hidden_omega_0 = 1.0

        self.pts_linears = torch.nn.ModuleList(
            [SineLayer(input_ch, W, omega_0=first_omega_0, is_first=unique_first)] +
            [SineLayer(W, W, omega_0=hidden_omega_0)
             if i not in self.skips else SineLayer(W + input_ch, W, omega_0=hidden_omega_0) for i in range(D - 1)]
        )

        self.sigma_linear = torch.nn.Linear(W, 1)

        if use_viewdirs:
            self.views_linear = SineLayer(3, W // 2, omega_0=first_omega_0)
            self.feature_linear = SineLayer(W, W // 2, omega_0=hidden_omega_0)
            self.feature_view_linears = torch.nn.ModuleList([SineLayer(W, W, omega_0=hidden_omega_0)])

        self.rgb_linear = torch.nn.Linear(W, 3)

    def update_fading_step(self, fading_step):
        # should be updated with the global step
        # e.g., update_fading_step(global_step - radiance_in_step)
        if fading_step >= 0:
            self.fading_step = fading_step

    def fading_wei_list(self):
        # try print(fading_wei_list()) for debug
        step_ratio = np.clip(float(self.fading_step) / float(max(1e-8, self.fading_fin_step)), 0, 1)
        ma = 1 + (self.D - 2) * step_ratio  # in range of 1 to self.D-1
        fading_wei_list = [np.clip(1 + ma - m, 0, 1) * np.clip(1 + m - ma, 0, 1) for m in range(self.D)]
        return fading_wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%0.03f" % (i, w_list[i]) for i in range(len(w_list)) if w_list[i] > 1e-8]
        print("; ".join(_str))

    def query_density_and_feature(self, input_pts: torch.Tensor, cond: torch.Tensor):
        h = input_pts
        h_layers = []
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)

            h_layers += [h]
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        # a sliding window (fading_wei_list) to enable deeper layers progressively
        if self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            h = 0
            for w, y in zip(fading_wei_list, h_layers):
                if w > 1e-8:
                    h = w * y + h

        sigma = self.sigma_linear(h)
        return torch.nn.functional.relu(sigma), h

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.query_density_and_feature(x, cond)[0]

    def forward(self, x, dirs, cond: torch.Tensor = None):
        sigma, h = self.query_density_and_feature(x, cond)

        if self.use_viewdirs:
            input_pts_feature = self.feature_linear(h)
            input_views_feature = self.views_linear(dirs)

            h = torch.cat([input_pts_feature, input_views_feature], -1)

            for i, l in enumerate(self.feature_view_linears):
                h = self.feature_view_linears[i](h)

        rgb = self.rgb_linear(h)
        # outputs = torch.cat([rgb, sigma], -1)

        return torch.sigmoid(rgb), sigma, {}


class SDFRadianceField(RadianceField):
    def __init__(self):
        super().__init__(True)

    @abc.abstractmethod
    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """ output sdf as [shape, 1] """
        pass

    @abc.abstractmethod
    def gradient(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """ output sdf gradient as [shape, 3] """
        pass

    # NeuS specific

    @abc.abstractmethod
    def deviation(self) -> torch.Tensor:
        """get inv_s as standard deviation"""
        pass

    def s_density(self, sdf: torch.Tensor, inv_s: torch.Tensor = None):
        if inv_s is None:
            inv_s = self.deviation()
        exp_sx = torch.exp(-sdf * inv_s)
        return inv_s * exp_sx / (1 + exp_sx) ** 2

    def opaque_density(self, sdf: torch.Tensor, inv_s: torch.Tensor = None):
        if inv_s is None:
            inv_s = self.deviation()
        rho = inv_s / (torch.exp(inv_s * sdf) + 1)  # phi_s(x) / Phi_s(x)
        return torch.clip(rho, max=self.solid_density)

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs):
        if kwargs.get('opaque', False):
            return self.opaque_density(self.sdf(x))
        return self.s_density(self.sdf(x))


class NeuS(SDFRadianceField):
    def __init__(self, D=8, W=256, input_ch=3, use_viewdirs=True, skips=(4,), n_features=256,
                 multires=0, multires_views=0, geometric_init=True, init_bias=0.5, bound=1.0, use_color_t=False,
                 output_s_density=True, init_variance=0.3, solid_density=5.0, fading_fin_step=1):
        super().__init__()

        self.input_ch = input_ch
        self.use_viewdirs = use_viewdirs > 0

        self.embed_fn, self.input_ch = positional_encoder(multires, input_dims=input_ch)
        self.embed_fn_views, self.input_ch_views = positional_encoder(multires_views)

        dims = [self.input_ch] + [W for _ in range(D - 1)] + [1 + n_features]
        self.num_layers = D
        self.skips = skips

        sdf_net = []
        for l in range(0, self.num_layers):
            if l + 1 in self.skips:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi / dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -init_bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2 / out_dim))
                elif multires > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2 / out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2 / out_dim))
            lin = torch.nn.utils.weight_norm(lin)

            sdf_net.append(lin)

        self.sdf_net = torch.nn.ModuleList(sdf_net)
        self.activation = torch.nn.Softplus(beta=100)

        self.use_color_t = use_color_t
        self.embed_t = None
        color_in_dims = 9 if self.use_viewdirs else 6
        if use_color_t:
            # self.embed_t, t_dim = positional_encoder(multires, input_dims=1)
            self.embed_t, t_dim = positional_encoder(0, input_dims=1)
            color_in_dims += t_dim
        dims = [n_features + color_in_dims] + [W for _ in range(4)] + [3]
        self.num_layers_color = 5

        color_net = []
        for l in range(0, self.num_layers_color):
            out_dim = dims[l + 1]
            lin = torch.nn.Linear(dims[l], out_dim)

            lin = torch.nn.utils.weight_norm(lin)

            color_net.append(lin)

        self.color_net = torch.nn.ModuleList(color_net)
        self.deviation_network = SingleVarianceNetwork(init_variance)

        self.bound = bound
        self.output_s_density = output_s_density
        self.solid_density = solid_density
        self.fading_step = 1
        self.fading_fin_step = max(1, fading_fin_step)
        self.multires = multires

    def deviation(self) -> torch.Tensor:
        return self.deviation_network()

    def update_fading_step(self, fading_step):
        if fading_step > 1:
            self.fading_step = fading_step

    def fading_wei_list(self):
        wei_list = [1.0]
        alpha = self.fading_step * self.multires / self.fading_fin_step
        for freq_n in range(self.multires):
            w_a = (1.0 - np.cos(np.pi * np.clip(alpha - freq_n, 0, 1))) * 0.5
            wei_list += [w_a, w_a]  # sin, cos
        return wei_list

    def print_fading(self):
        w_list = self.fading_wei_list()
        _str = ["h%d:%.3f" % (i * 3, w_list[i])
                for i in range(len(w_list)) if 1e-5 < w_list[i] < 1 - 1e-5]
        print("; ".join(_str))

    def forward_sdf(self, inputs):
        inputs = inputs / self.bound  # map to [0, 1]
        # inputs = inputs * 2 / self.bound - 1.0  # map to [-1, 1]
        inputs = self.embed_fn(inputs)

        if self.multires > 0 and self.fading_fin_step > self.fading_step:
            fading_wei_list = self.fading_wei_list()
            pts = torch.zeros_like(inputs)
            for i, wei in enumerate(fading_wei_list):
                if wei > 1e-8:
                    bgn = i * 3
                    end = bgn + 3  # full fading
                    pts[..., bgn: end] = inputs[..., bgn: end] * wei
            inputs = pts

        x = inputs
        for l in range(0, self.num_layers):
            lin = self.sdf_net[l]

            if l in self.skips:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 1:
                x = self.activation(x)
        return x

    def sdf(self, x):
        return self.forward_sdf(x)[..., :1]

    def forward_with_gradient(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            output = self.forward_sdf(x)
            y = output[..., :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        return output, gradients

    def gradient(self, x, **kwargs):
        return self.forward_with_gradient(x)[1]

    def forward_color(self, points, normals, view_dirs, color_t, feature_vectors):
        cat_list = [points]
        if self.use_viewdirs:
            view_dirs = self.embed_fn_views(view_dirs)
            cat_list.append(view_dirs)
        if self.use_color_t:
            cat_list.append(self.embed_t(color_t))
        cat_list += [normals, feature_vectors]
        rendering_input = torch.cat(cat_list, dim=-1)

        x = rendering_input
        for l in range(0, self.num_layers_color):
            lin = self.color_net[l]

            x = lin(x)
            if l < self.num_layers_color - 1:
                x = torch.nn.functional.relu(x)

        x = torch.sigmoid(x)
        return x

    def forward(self, input_pts, input_views, cond: torch.Tensor = None):
        sdf_nn_output, gradients = self.forward_with_gradient(input_pts)
        sdf = sdf_nn_output[..., :1]
        feature_vectors = sdf_nn_output[..., 1:]

        # x, n, v, z in IDR
        sampled_color = self.forward_color(input_pts, gradients, input_views, cond, feature_vectors)
        inv_s = self.deviation_network()

        if self.output_s_density:
            sigma = self.s_density(sdf, inv_s)
            # sigma = self.opaque_density(sdf, inv_s)
        else:
            sigma = None
        return sampled_color, sigma, {
            'sdf': sdf,
            'gradients': gradients,
            'inv_s': inv_s
        }


class HybridRadianceField(RadianceField):
    def __init__(self, static_model: RadianceField, dynamic_model: RadianceField):
        super().__init__(static_model.output_sdf)
        self.static_model = static_model
        self.dynamic_model = dynamic_model

    def update_fading_step(self, fading_step: int):
        self.static_model.update_fading_step(fading_step)
        self.dynamic_model.update_fading_step(fading_step)

    def print_fading(self):
        print('static: ', end='')
        self.static_model.print_fading()
        print('dynamic: ', end='')
        self.dynamic_model.print_fading()

    def query_density(self, x: torch.Tensor, cond: torch.Tensor = None, **kwargs) -> torch.Tensor:
        s_static = self.static_model.query_density(x[..., :3], cond, **kwargs)
        s_dynamic = self.dynamic_model.query_density(x, cond)
        return s_static + s_dynamic

    def forward(self, x: torch.Tensor, dirs: torch.Tensor | None, cond: torch.Tensor = None):
        rgb_s, sigma_s, extra_s = self.static_model.forward(x[..., :3], dirs, cond)
        rgb_d, sigma_d, extra_d = self.dynamic_model.forward(x, dirs, cond)
        return self.merge_result(self.output_sdf, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d)

    @staticmethod
    def merge_result(output_sdf: bool, rgb_s, sigma_s, extra_s, rgb_d, sigma_d, extra_d):
        if output_sdf:
            sigma = sigma_d
            rgb = rgb_d
        else:  # does alpha blend, when delta -> 0
            sigma = sigma_s + sigma_d
            rgb = (rgb_s * sigma_s + rgb_d * sigma_d) / (sigma + 1e-6)

        extra_s |= {
            'rgb_s': rgb_s,
            'rgb_d': rgb_d,
            'sigma_s': sigma_s,
            'sigma_d': sigma_d,
        }
        if len(extra_d) > 0:
            extra_s['dynamic'] = extra_d
        return rgb, sigma, extra_s


# ======================================== PINFRender ========================================
from typing import NamedTuple, Callable, Optional


class Rays(NamedTuple):
    origins: torch.Tensor
    viewdirs: torch.Tensor

    def foreach(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        return Rays(fn(self.origins), fn(self.viewdirs))

    def to(self, device):
        return Rays(self.origins.to(device), self.viewdirs.to(device))


class NeRFOutputs:
    def __init__(self, rgb_map: torch.Tensor, depth_map: torch.Tensor | None, acc_map: torch.Tensor, **kwargs):
        """
        Args:
            rgb_map: [n_rays, 3]. Estimated RGB color of a ray.
            depth_map: [n_rays]. Depth map. Optional.
            acc_map: [n_rays]. Sum of weights along each ray.
        """
        self.rgb = rgb_map
        self.depth = depth_map
        self.acc = acc_map
        self.extras = kwargs

    def as_tuple(self):
        return self.rgb, self.depth, self.acc, self.extras

    @staticmethod
    def merge(outputs: list["NeRFOutputs"], shape=None, skip_extras=False) -> "NeRFOutputs":
        """Merge list of outputs into one
        Args:
            outputs: Outputs from different batches.
            shape: If not none, reshape merged outputs' first dimension
            skip_extras: Ignore extras when merging, used for merging coarse outputs
        """
        if len(outputs) == 1:  # when training
            return outputs[0]
        extras = {}
        if not skip_extras:
            keys = outputs[0].extras.keys()  # all extras must have same keys
            extras = {k: [] for k in keys}
            for output in outputs:
                for k in keys:
                    extras[k].append(output.extras[k])
            for k in extras:
                assert isinstance(extras[k][0], (torch.Tensor, NeRFOutputs)), \
                    "All extras must be either torch.Tensor or NeRFOutputs when merging"
                if isinstance(extras[k][0], NeRFOutputs):
                    extras[k] = NeRFOutputs.merge(extras[k], shape)  # recursive merging
                elif extras[k][0].dim() == 0:
                    extras[k] = torch.tensor(extras[k]).mean()  # scalar value, reduce to avg
                else:
                    extras[k] = torch.cat(extras[k])

        ret = NeRFOutputs(
            torch.cat([out.rgb for out in outputs]),
            torch.cat([out.depth for out in outputs]) if outputs[0].depth is not None else None,
            torch.cat([out.acc for out in outputs]),
            **extras
        )
        if shape is not None:
            ret.rgb = ret.rgb.reshape(*shape, 3)
            ret.depth = ret.depth.reshape(shape) if ret.depth is not None else None
            ret.acc = ret.acc.reshape(shape)
            for k in ret.extras:
                if isinstance(ret.extras[k], torch.Tensor) and ret.extras[k].dim() > 0:
                    ret.extras[k] = torch.reshape(ret.extras[k], [*shape, *ret.extras[k].shape[1:]])
        return ret

    def add_background(self, background: torch.Tensor):
        """Add background to rgb output
        Args:
            background: scalar or image
        """
        self.rgb = self.rgb + background * (1.0 - self.acc[..., None])
        for v in self.extras.values():
            if isinstance(v, NeRFOutputs):
                v.add_background(background)


def sdf2alpha(sdf: torch.Tensor, gradients: torch.Tensor, inv_s: torch.Tensor,
              rays_d: torch.Tensor, dists: torch.Tensor, cos_anneal_ratio: float = 1.0):
    true_cos = (rays_d[..., None, :] * gradients).sum(-1, keepdim=True)
    iter_cos = -(torch.nn.functional.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                 torch.nn.functional.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
    sdf_diff = iter_cos * dists.reshape(*iter_cos.shape) * 0.5

    # Estimate signed distances at section points
    # estimated_next_sdf = sdf + iter_cos * dists.reshape(*iter_cos.shape) * 0.5
    # estimated_prev_sdf = sdf - iter_cos * dists.reshape(*iter_cos.shape) * 0.5

    prev_cdf = torch.sigmoid((sdf - sdf_diff) * inv_s)
    next_cdf = torch.sigmoid((sdf + sdf_diff) * inv_s)

    # p = prev_cdf - next_cdf
    # c = prev_cdf
    # alpha = ((p + 1e-5) / (c + 1e-5)).reshape(*sdf.shape[:-1]).clip(0.0, 1.0)
    alpha = (1.0 - next_cdf / (prev_cdf + 1e-5)).clip(0.0, 1.0)
    return alpha


def weighted_sum_of_samples(wei_list: list[torch.Tensor], content: list[torch.Tensor] | torch.Tensor | None):
    if isinstance(content, list):  # list of [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * ct, dim=-2) for weights, ct in zip(wei_list, content)]

    elif content is not None:  # [n_rays, n_samples, dim]
        return [torch.sum(weights[..., None] * content, dim=-2) for weights in wei_list]

    return [torch.sum(weights, dim=-1) for weights in wei_list]


def raw2outputs(raw, z_vals, rays_d, mask=None, cos_anneal_ratio=1.0) -> tuple[NeRFOutputs, torch.Tensor]:
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: returned result of RadianceField: rgb, sigma, extra. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        mask: [num_rays, num_samples]. aabb masking
        cos_anneal_ratio: float.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [n_rays, n_samples]
    dists = torch.cat([dists, dists[..., -1:]], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    def sigma2alpha(sigma: torch.Tensor):  # [n_rays, n_samples, 1] -> [n_rays, n_samples, 1]
        if mask is not None:
            sigma = sigma * mask
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # [n_rays, n_samples]
        return alpha

    extra: dict = raw[2]
    gradients = None
    if 'sdf' in extra:
        sdf = extra['sdf']
        sdf = torch.where(mask, sdf, 1e10)  # must larger than max dists
        gradients = extra['gradients']
        inv_s = extra['inv_s']
        alpha_s = sdf2alpha(sdf, gradients, inv_s, rays_d, dists, cos_anneal_ratio)

        norm = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
        gradients = gradients / norm
        extra['eikonal_loss'] = (mask * (norm - 1.0) ** 2).mean()
        extra['sdf'] = sdf
        # extra['gradients'] = gradients
        extra.pop('gradients')
        alpha_list = [sigma2alpha(extra['sigma_d']), alpha_s.squeeze(-1)]
        color_list = [extra['rgb_d'], extra['rgb_s']]

    elif 'sigma_s' in extra:
        alpha_list = [sigma2alpha(extra['sigma_d']), sigma2alpha(extra['sigma_s'])]
        color_list = [extra['rgb_d'], extra['rgb_s']]

    else:
        # shortcut for single model
        alpha = sigma2alpha(raw[1])
        rgb = raw[0]
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [n_rays, 3]
        # depth_map = torch.sum(weights * z_vals, -1)
        depth_map = None  # unused
        acc_map = torch.sum(weights, -1)
        return NeRFOutputs(rgb_map, depth_map, acc_map), weights

    for key in 'rgb_s', 'rgb_d', 'dynamic':
        extra.pop(key, None)

    dens = 1.0 - torch.stack(alpha_list, dim=-1)  # [n_rays, n_samples, 2]
    dens = torch.cat([dens, torch.prod(dens, dim=-1, keepdim=True)], dim=-1) + 1e-9  # [n_rays, n_samples, 3]
    Ti_all = torch.cumprod(dens, dim=-2) / dens  # [n_rays, n_samples, 3], accu along samples, exclusive
    weights_list = [alpha * Ti_all[..., -1] for alpha in alpha_list]  # a list of [n_rays, n_samples]

    rgb_map = sum(weighted_sum_of_samples(weights_list, color_list))  # [n_rays, 3]
    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = sum(weighted_sum_of_samples(weights_list, None))  # [n_rays]

    # Estimated depth map is expected distance.
    # Disparity map is inverse depth.
    # depth_map = sum(weighted_sum_of_samples(weights_list, z_vals[..., None]))  # [n_rays]
    depth_map = None
    # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
    # alpha * Ti
    weights = weights_list[0]  # [n_rays, n_samples]

    if len(alpha_list) > 1:  # hybrid model
        self_weights_list = [alpha_list[alpha_i] * Ti_all[..., alpha_i] for alpha_i in
                             range(len(alpha_list))]  # a list of [n_rays, n_samples]
        rgb_map_stack = weighted_sum_of_samples(self_weights_list, color_list)
        acc_map_stack = weighted_sum_of_samples(self_weights_list, None)

        if gradients is not None:
            extra['grad_map'] = weighted_sum_of_samples(self_weights_list[1:], gradients)[0]

        # assume len(alpha_list) == 2 for hybrid model
        extra['dynamic'] = NeRFOutputs(rgb_map_stack[0], None, acc_map_stack[0])
        extra['static'] = NeRFOutputs(rgb_map_stack[1], None, acc_map_stack[1])

    return NeRFOutputs(rgb_map, depth_map, acc_map, **extra), weights


def mask_from_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    pts = pts[..., :3]
    inside = torch.logical_and(torch.less_equal(aabb[:3], pts), torch.less_equal(pts, aabb[3:]))
    return torch.logical_and(torch.logical_and(inside[..., 0], inside[..., 1]), inside[..., 2]).unsqueeze(-1)


def get_warped_pts(vel_model, orig_pts: torch.Tensor, fading: float, mod: str = "rand") -> torch.Tensor:
    # mod, "rand", "forw", "back", "none"
    if (mod == "none") or (vel_model is None):
        return orig_pts

    orig_pos, orig_t = torch.split(orig_pts, [3, 1], -1)

    with torch.no_grad():
        _vel = vel_model(orig_pts)
    # _vel.shape, [n_rays, n_samples(+n_importance), 3]
    if mod == "rand":
        # random_warpT = np.random.normal(0.0, 0.6, orig_t.get_shape().as_list())
        # random_warpT = np.random.uniform(-3.0, 3.0, orig_t.shape)
        random_warpT = torch.rand_like(orig_t) * 6.0 - 3.0  # [-3,3]
    else:
        random_warpT = 1.0 if mod == "back" else (-1.0)  # back
    # mean and standard deviation: 0.0, 0.6, so that 3sigma < 2, train +/- 2*delta_T
    random_warpT = random_warpT * fading
    # random_warpT = torch.Tensor(random_warpT)

    warp_t = orig_t + random_warpT
    warp_pos = orig_pos + _vel * random_warpT
    warp_pts = torch.cat([warp_pos, warp_t], dim=-1)
    warp_pts = warp_pts.detach()  # stop gradiant

    return warp_pts


def get_warped_raw(model: RadianceField, vel_model, warp_mod, warp_fading_dt, pts, dirs):
    if warp_mod == "none" or None in [vel_model, warp_fading_dt]:
        # no warping
        return model.forward(pts, dirs, pts[..., -1:])

    warp_pts = get_warped_pts(vel_model, pts, warp_fading_dt, warp_mod)
    if not isinstance(model, HybridRadianceField):
        return model.forward(warp_pts, dirs)

    raw_s = model.static_model.forward(pts[..., :3], dirs, pts[..., -1:])
    raw_d = model.dynamic_model.forward(warp_pts, dirs)
    raw = HybridRadianceField.merge_result(model.output_sdf, *raw_s, *raw_d)
    return raw


def attach_time(pts: torch.Tensor, t: float):
    return torch.cat([pts, torch.tensor(t, dtype=pts.dtype, device=pts.device).expand(*pts.shape[:-1], 1)], dim=-1)


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    device = weights.get_device()
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf.detach(), u, right=True)

    below = torch.max(torch.zeros_like(inds - 1, device=device), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds, device=device), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def up_sample(sdf_network: SDFRadianceField, rays_o, rays_d, z_vals, sdf, n_importance, inv_s,
              last=False, inside_sphere_test=True):
    batch_size, n_samples = z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
    sdf = sdf.reshape(batch_size, n_samples)
    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

    # ----------------------------------------------------------------------------------------------------------
    # Use min value of [ cos, prev_cos ]
    # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
    # robust when meeting situations like below:
    #
    # SDF
    # ^
    # |\          -----x----...
    # | \        /
    # |  x      x
    # |---\----/-------------> 0 level
    # |    \  /
    # |     \/
    # |
    # ----------------------------------------------------------------------------------------------------------
    prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=sdf.device), cos_val[:, :-1]], dim=-1)
    cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
    cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
    cos_val = cos_val.clip(-1e3, 0.0)

    if inside_sphere_test:
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        cos_val = cos_val * inside_sphere

    dist = (next_z_vals - prev_z_vals)
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]

    new_z_vals = sample_pdf(z_vals, weights, n_importance, det=True).detach()

    # cat_z_vals
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_sdf = sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        sdf = torch.cat([sdf, new_sdf], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, sdf


class PINFRenderer:
    def __init__(self,
                 model: RadianceField,
                 prop_model: RadianceField | None,
                 n_samples: int,
                 n_importance: int = 0,
                 near: torch.Tensor | float = 0.0,
                 far: torch.Tensor | float = 1.0,
                 perturb: bool = False,
                 vel_model=None,
                 warp_fading_dt=None,
                 warp_mod="rand",
                 aabb: torch.Tensor = None,
                 ):
        """Volumetric rendering.
        Args:
          model: Model for predicting RGB and density at each pointin space.
          n_samples: int. Number of different times to sample along each ray.
          n_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          warp_fading_dt, to train nearby frames with flow-based warping, fading*delt_t
        """
        self.model = model
        self.prop_model = prop_model
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far
        self.perturb = perturb
        self.vel_model = vel_model
        self.warp_fading_dt = warp_fading_dt
        self.warp_mod = warp_mod
        self.aabb = aabb
        self.cos_anneal_ratio = 1.0

    def run(self, rays: Rays, rays_t: float | None, near: torch.Tensor, far: torch.Tensor,
            ret_raw: bool = False, perturb: bool = None, ignore_vel: bool = False
            ) -> NeRFOutputs:

        n_samples = self.n_samples
        n_importance = self.n_importance
        model = self.model
        prop_model = self.prop_model if self.prop_model is not None else model
        vel_model = None if ignore_vel else self.vel_model
        warp_mod = self.warp_mod
        warp_fading_dt = self.warp_fading_dt
        if perturb is None:
            perturb = self.perturb

        rays_o, rays_d = rays  # [n_rays, 3] each
        n_rays = rays_o.shape[0]

        t_vals = torch.linspace(0., 1., steps=n_samples, device=rays_o.device)
        z_vals = near + (far - near) * t_vals
        z_vals = z_vals.expand([n_rays, n_samples])

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [n_rays, n_samples, 3]
        if rays_t is not None:
            pts = attach_time(pts, rays_t)
        viewdirs = rays_d[..., None, :].expand(*pts.shape[:-1], -1)

        # raw = run_network(pts)
        raw = get_warped_raw(prop_model, vel_model, warp_mod, warp_fading_dt, pts, viewdirs)
        mask = mask_from_aabb(pts, self.aabb)
        outputs, weights = raw2outputs(raw, z_vals, rays_d, mask, cos_anneal_ratio=self.cos_anneal_ratio)

        if n_importance > 0:
            outputs_0 = outputs

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            weights = weights[..., 1:-1].detach()

            if self.model.output_sdf:
                # mixed up sampling for hybrid neus model
                assert isinstance(self.model, HybridRadianceField)
                sdf_model = self.model.static_model
                assert isinstance(sdf_model, SDFRadianceField)

                sdf_samples = self.n_importance // 2
                z_samples = sample_pdf(z_vals_mid, weights, self.n_importance - sdf_samples, det=not perturb)

                with torch.no_grad():
                    sdf = sdf_model.sdf(pts[..., :3])
                    up_sample_steps = 4  # TODO config up_sample_steps in NeuS
                    for i in range(up_sample_steps):
                        z_vals, sdf = up_sample(
                            sdf_model, rays_o, rays_d, z_vals, sdf,
                            sdf_samples // up_sample_steps,
                            inv_s=64 * 2 ** i,
                            last=(i + 1 == up_sample_steps), inside_sphere_test=False)

            else:
                z_samples = sample_pdf(z_vals_mid, weights, self.n_importance, det=not perturb)

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            # [n_rays, n_samples + n_importance, 3]
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
            if rays_t is not None:
                pts = attach_time(pts, rays_t)
            viewdirs = rays_d[..., None, :].expand(*pts.shape[:-1], -1)

            raw = get_warped_raw(model, vel_model, warp_mod, warp_fading_dt, pts, viewdirs)
            mask = mask_from_aabb(pts, self.aabb)
            outputs, _ = raw2outputs(raw, z_vals, rays_d, mask, cos_anneal_ratio=self.cos_anneal_ratio)
            outputs.extras['coarse'] = outputs_0

        if not ret_raw:
            outputs.extras = {k: outputs.extras[k] for k in outputs.extras
                              if k.endswith('map') or isinstance(outputs.extras[k], NeRFOutputs)}

        return outputs

    def render(self, rays_o, rays_d, chunk=1024 * 32,
               timestep: float = None, background=None,
               **kwargs) -> NeRFOutputs:
        """Render rays
        Args:
          H: int. Height of image in pixels.
          W: int. Width of image in pixels.
          focal: float. Focal length of pinhole camera.
          chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.
          rays: array of shape [2, batch_size, 3]. Ray origin and direction for
            each example in batch.
          c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
          near: float or array of shape [batch_size]. Nearest distance for a ray.
          far: float or array of shape [batch_size]. Farthest distance for a ray.
          c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
           camera while using other c2w argument for viewing directions.
        Returns:
          rgb_map: [batch_size, 3]. Predicted RGB values for rays.
          disp_map: [batch_size]. Disparity map. Inverse of depth.
          acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
          extras: dict with everything returned by render_rays().
        """
        shape = rays_d.shape[:-1]  # [..., 3]

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        rays = Rays(rays_o, rays_d)

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])

        # Render and reshape
        ret_list = []
        for i in range(0, rays_o.shape[0], chunk):
            rays_chunk = rays.foreach(lambda t: t[i: i + chunk])
            ret = self.run(rays_chunk, timestep, near=near[i: i + chunk], far=far[i: i + chunk], **kwargs)
            ret_list.append(ret)

        output = NeRFOutputs.merge(ret_list, shape)
        if background is not None:
            output.add_background(background)

        return output


# ======================================== PINFRender ========================================

def model_fading_update(model: RadianceField, prop_model: RadianceField | None, vel_model: SIREN_vel | None,
                        global_step, vel_delay):
    model.update_fading_step(global_step)
    if prop_model is not None:
        prop_model.update_fading_step(global_step)
    if vel_model is not None:
        vel_model.update_fading_step(global_step - vel_delay)


# VGG Tool, https://github.com/crowsonkb/style-transfer-pytorch/
# set pooling = 'max'
class VGGFeatures(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = sorted(set(layers))

        # The PyTorch pre-trained VGG-19 expects sRGB inputs in the range [0, 1] which are then
        # normalized according to this transform, unlike Simonyan et al.'s original model.
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])

        # The PyTorch pre-trained VGG-19 has different parameters from Simonyan et al.'s original
        # model.
        self.model = torchvision.models.vgg19(pretrained=True).features[:self.layers[-1] + 1]

        # Reduces edge artifacts.
        self.model[0] = self._change_padding_mode(self.model[0], 'replicate')

        self.model.eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _change_padding_mode(conv, padding_mode):
        new_conv = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding,
                                   padding_mode=padding_mode)
        with torch.no_grad():
            new_conv.weight.copy_(conv.weight)
            new_conv.bias.copy_(conv.bias)
        return new_conv

    @staticmethod
    def _get_min_size(layers):
        last_layer = max(layers)
        min_size = 1
        for layer in [4, 9, 18, 27, 36]:
            if last_layer < layer:
                break
            min_size *= 2
        return min_size

    def forward(self, input, layers=None):
        # input shape, b,3,h,w
        layers = self.layers if layers is None else sorted(set(layers))
        h, w = input.shape[2:4]
        min_size = self._get_min_size(layers)
        if min(h, w) < min_size:
            raise ValueError(f'Input is {h}x{w} but must be at least {min_size}x{min_size}')
        feats = {'input': input}
        norm_in = torch.stack([self.normalize(input[_i]) for _i in range(input.shape[0])], dim=0)
        # input = self.normalize(input)
        for i in range(max(layers) + 1):
            norm_in = self.model[i](norm_in)
            if i in layers:
                feats[i] = norm_in
        return feats


# VGG Loss Tool
class VGGLossTool(object):
    def __init__(self, device):
        # The default content and style layers in Gatys et al. (2015):
        #   content_layers = [22], 'relu4_2'
        #   style_layers = [1, 6, 11, 20, 29], relu layers: [ 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        # We use [5, 10, 19, 28], conv layers before relu: [ 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.layer_list = [5, 10, 19, 28]
        self.layer_names = [
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.device = device

        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.vggmodel = VGGFeatures(self.layer_list).to(device)

    def feature_norm(self, feature):
        # feature: b,h,w,c
        feature_len = torch.sqrt(torch.sum(torch.square(feature), dim=-1, keepdim=True) + 1e-12)
        norm = feature / feature_len
        return norm

    def cos_sim(self, a, b):
        cos_sim_ab = torch.sum(a * b, dim=-1)
        # cosine similarity, -1~1, 1 best
        cos_sim_ab_score = 1.0 - torch.mean(cos_sim_ab)  # 0 ~ 2, 0 best
        return cos_sim_ab_score

    def compute_cos_loss(self, img, ref):
        # input img, ref should be in range of [0,1]
        input_tensor = torch.stack([ref, img], dim=0)

        input_tensor = input_tensor.permute((0, 3, 1, 2))
        # print(input_tensor.shape)
        _feats = self.vggmodel(input_tensor, layers=self.layer_list)

        # Initialize the loss
        loss = []
        # Add loss
        for layer_i, layer_name in zip(self.layer_list, self.layer_names):
            cur_feature = _feats[layer_i]
            reference_features = self.feature_norm(cur_feature[0, ...])
            img_features = self.feature_norm(cur_feature[1, ...])

            feature_metric = self.cos_sim(reference_features, img_features)
            loss += [feature_metric]
        return loss


def vgg_sample(vgg_strides: int, num_rays: int, frame: torch.Tensor, bg_color: torch.Tensor, dw: int = None,
               steps: int = None):
    if steps is None:
        strides = vgg_strides + np.random.randint(-1, 2)  # args.vgg_strides(+/-)1 or args.vgg_strides
    else:
        strides = vgg_strides + steps % 3 - 1
    H, W = frame.shape[:2]
    if dw is None:
        dw = max(20, min(40, int(np.sqrt(num_rays))))
    vgg_min_border = 10
    strides = min(strides, min(H - vgg_min_border, W - vgg_min_border) / dw)
    strides = int(strides)

    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W), indexing='ij'),
                         dim=-1).to(frame.device)  # (H, W, 2)
    target_grey = torch.mean(torch.abs(frame - bg_color), dim=-1, keepdim=True)  # (H, W, 1)
    img_wei = coords.to(torch.float32) * target_grey
    center_coord = torch.sum(img_wei, dim=(0, 1)) / torch.sum(target_grey)
    center_coord = center_coord.cpu().numpy()
    # add random jitter
    random_R = dw * strides / 2.0
    # mean and standard deviation: center_coord, random_R/3.0, so that 3sigma < random_R
    random_x = np.random.normal(center_coord[1], random_R / 3.0) - 0.5 * dw * strides
    random_y = np.random.normal(center_coord[0], random_R / 3.0) - 0.5 * dw * strides

    offset_w = int(min(max(vgg_min_border, random_x), W - dw * strides - vgg_min_border))
    offset_h = int(min(max(vgg_min_border, random_y), H - dw * strides - vgg_min_border))

    coords_crop = coords[offset_h:offset_h + dw * strides:strides, offset_w:offset_w + dw * strides:strides, :]
    return coords_crop, dw


def fade_in_weight(step, start, duration):
    return min(max((float(step) - start) / duration, 0.0), 1.0)


# Ghost Density Loss Tool
def ghost_loss_func(out: NeRFOutputs, bg: torch.Tensor, scale: float = 4.0):
    ghost_mask = torch.mean(torch.square(out.rgb - bg), -1)
    # ghost_mask = torch.sigmoid(ghost_mask*-1.0) + den_penalty # (0 to 0.5) + den_penalty
    ghost_mask = torch.exp(ghost_mask * -scale)
    ghost_alpha = ghost_mask * out.acc
    return torch.mean(torch.square(ghost_alpha))


# ======================================== Data Loader ========================================
import os
import imageio
import cv2
import json
from torch.utils.data import Dataset


def trans_t(t: float):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_phi(phi: float):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def rot_theta(th: float):
    return np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]], dtype=np.float32)


def pose_spherical(theta: float, phi: float, radius: float, rotZ=True, center: np.ndarray = None):
    # spherical, rotZ=True: theta rotate around Z; rotZ=False: theta rotate around Y
    # center: additional translation, normally the center coord.
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    if rotZ:  # swap yz, and keep right-hand
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32) @ c2w

    if center is not None:
        c2w[:3, 3] += center
    return c2w


def intrinsics_from_hwf(H: int, W: int, focal: float):
    return np.array([
        [focal, 0, 0.5 * W],
        [0, focal, 0.5 * H],
        [0, 0, 1]
    ], dtype=np.float32)


class VideoData:
    def __init__(self, args: dict | None, basedir: str = '', half_res: str = None):
        if args is None:
            self.delta_t = 1.0
            self.transform_matrix = np.empty(0)
            self.frames = np.empty(0)
            self.focal = 0.0
            return

        filename = os.path.join(basedir, args['file_name'])
        meta = imageio.v3.immeta(filename)
        reader = imageio.imiter(filename)

        frame_rate = args.get('frame_rate', meta['fps'])
        frame_num = args.get('frame_num')
        if not np.isfinite(frame_num):
            frame_num = meta['nframes']
            if not np.isfinite(frame_num):
                frame_num = meta['duration'] * meta['fps']
            frame_num = round(frame_num)

        self.delta_t = 1.0 / frame_rate
        if 'transform_matrix' in args:
            self.transform_matrix = np.array(args['transform_matrix'], dtype=np.float32)
        else:
            self.transform_matrix = np.array(args['transform_matrix_list'], dtype=np.float32)

        frames = tuple(reader)[:frame_num]
        H, W = frames[0].shape[:2]
        if half_res == 'half':
            H //= 2
            W //= 2
        elif half_res == 'quarter':
            H //= 4
            W //= 4
        elif half_res is not None:
            if half_res != 'normal':
                print("Unsupported half_res value", half_res)
            half_res = None

        if half_res is not None:
            frames = [cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) for frame in frames]
        self.frames: np.ndarray = np.float32(frames) / 255.0
        self.focal = float(0.5 * self.frames.shape[2] / np.tan(0.5 * args['camera_angle_x']))

    def c2w(self, frame: int = None) -> np.ndarray:
        if self.transform_matrix.ndim == 2 or frame is None:
            return self.transform_matrix
        return self.transform_matrix[frame]

    def intrinsics(self):
        return intrinsics_from_hwf(self.frames.shape[1], self.frames.shape[2], self.focal)

    def __len__(self) -> int:
        return self.frames.shape[0]


class PINFFrameDataBase:
    def __init__(self):
        # placeholders
        self.voxel_tran: np.ndarray | None = None
        self.voxel_scale: np.ndarray | None = None
        self.videos: dict[str, list[VideoData]] = {}
        self.t_info: np.ndarray | None = None
        self.render_poses: np.ndarray | None = None
        self.render_timesteps: np.ndarray | None = None
        self.bkg_color: np.ndarray | None = None
        self.near, self.far = 0.0, 1.0


class PINFFrameData(PINFFrameDataBase):
    def __init__(self, basedir: str, half_res: str | bool = None, normalize_time: bool = False,
                 apply_tran: bool = False, **kwargs):
        super().__init__()
        with open(os.path.join(basedir, 'info.json'), 'r') as fp:
            # read render settings
            meta = json.load(fp)
        near = float(meta['near'])
        far = float(meta['far'])
        radius = (near + far) * 0.5
        phi = float(meta['phi'])
        rotZ = (meta['rot'] == 'Z')
        r_center = np.float32(meta['render_center'])
        bkg_color = np.float32(meta['frame_bkg_color'])
        if isinstance(half_res, bool):  # compatible with nerf
            half_res = 'half' if half_res else None

        # read scene data
        voxel_tran = np.float32(meta['voxel_matrix'])
        voxel_tran = np.stack([voxel_tran[:, 2], voxel_tran[:, 1], voxel_tran[:, 0], voxel_tran[:, 3]],
                              axis=1)  # swap_zx
        voxel_scale = np.broadcast_to(meta['voxel_scale'], [3]).astype(np.float32)

        if apply_tran:
            voxel_tran[:3, :3] *= voxel_scale[0]
            scene_tran = np.linalg.inv(voxel_tran)
            voxel_tran = np.eye(4, dtype=np.float32)
            voxel_scale /= voxel_scale[0]
            near, far = 0.1, 2.0  # TODO apply conversion

        else:
            scene_tran = None

        self.voxel_tran: np.ndarray = voxel_tran
        self.voxel_scale: np.ndarray = voxel_scale

        self.videos: dict[str, list[VideoData]] = {
            'train': [],
            'test': [],
            'val': [],
        }

        # read video frames
        # all videos should be synchronized, having the same frame_rate and frame_num
        for s in ('train', 'val', 'test'):
            video_list = meta[s + '_videos'] if (s + '_videos') in meta else []

            for train_video in video_list:
                video = VideoData(train_video, basedir, half_res=half_res)
                self.videos[s].append(video)

            if len(video_list) == 0:
                self.videos[s] = self.videos['train'][:1]

        self.videos['test'] += self.videos['val']  # val vid not used for now
        self.videos['test'] += self.videos['train']  # for test
        video = self.videos['train'][0]
        # assume identical frame rate and length
        if normalize_time:
            self.t_info = np.float32([0.0, 1.0, 1.0 / len(video)])
        else:
            self.t_info = np.float32([0.0, video.delta_t * len(video), video.delta_t])  # min t, max t, delta_t

        # set render settings:
        sp_n = 40  # an even number!
        sp_poses = [
            pose_spherical(angle, phi, radius, rotZ, r_center)
            for angle in np.linspace(-180, 180, sp_n + 1)[:-1]
        ]

        if scene_tran is not None:
            for vk in self.videos:
                for video in self.videos[vk]:
                    video.transform_matrix = scene_tran @ video.transform_matrix
            sp_poses = [scene_tran @ pose for pose in sp_poses]

        self.render_poses = np.stack(sp_poses, 0)  # [sp_poses[36]]*sp_n, for testing a single pose
        self.render_timesteps = np.linspace(self.t_info[0], self.t_info[1], num=sp_n).astype(np.float32)
        self.bkg_color = bkg_color
        self.near, self.far = near, far


# use multiple videos as dataset
class PINFDataset:
    def __init__(self, base: PINFFrameDataBase, split: str = 'train'):
        super().__init__()
        self.base = base
        self.videos = self.base.videos[split]

    def __len__(self):
        return len(self.videos) * len(self.videos[0])

    def get_video_and_frame(self, item: int) -> tuple[VideoData, int]:
        vi, fi = divmod(item, len(self.videos[0]))
        video = self.videos[vi]
        return video, fi


class NeRFDataset(Dataset):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, item: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
        """returns image(optional for test), pose, intrinsics(optional for test), and extras"""
        pass

    # poses for prediction, usually given by pose_spherical
    def predict_poses(self):
        render_poses = getattr(self, 'render_poses', None)
        return render_poses if isinstance(render_poses, np.ndarray) else None

    # generate predictor from render_poses
    def predictor(self) -> Optional["NeRFPredictor"]:
        poses = self.predict_poses()
        return NeRFPredictor(poses) if poses is not None else None


class NeRFPredictor(NeRFDataset):
    """Given poses for prediction (usually generated from pose_spherical), used for test"""

    def __init__(self, poses: np.ndarray, extra_fn: Callable[[int], dict] = lambda _: {}):
        super().__init__()
        self.poses = poses
        self.extra_fn = extra_fn

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, item):
        # image, intrinsics unavailable
        return None, self.poses[item], None, self.extra_fn(item)

    def predict_poses(self):
        return self.poses

    def predictor(self):
        return self


class NeRFDataset(Dataset):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, item: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None, dict]:
        """returns image(optional for test), pose, intrinsics(optional for test), and extras"""
        pass

    # poses for prediction, usually given by pose_spherical
    def predict_poses(self):
        render_poses = getattr(self, 'render_poses', None)
        return render_poses if isinstance(render_poses, np.ndarray) else None

    # generate predictor from render_poses
    def predictor(self) -> Optional["NeRFPredictor"]:
        poses = self.predict_poses()
        return NeRFPredictor(poses) if poses is not None else None


# use a validate/test video for validation/testing
class PINFTestDataset(NeRFDataset):
    def __init__(self, base: PINFFrameDataBase, split: str = 'test', video_id: int = 0,
                 bkg_color: np.ndarray = None, skip: int = 1):
        super().__init__()
        self.base = base
        self.video = self.base.videos[split][video_id]
        self.bkg_color = base.bkg_color if bkg_color is None else bkg_color
        self.skip = skip

    def __len__(self):
        return len(self.video) // self.skip

    # use poses from test video, have gt
    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        item = item * self.skip
        frame = self.video.frames[item]
        c2w = self.video.c2w(item)
        timestep = item * self.base.t_info[-1]
        return frame, c2w, self.video.intrinsics(), {
            "timestep": timestep,
        }

    def predict_poses(self):
        return self.base.render_poses

    def predictor(self):
        return NeRFPredictor(self.base.render_poses, lambda x: {"timestep": self.base.render_timesteps[x]})


# ======================================== Data Loader ========================================

def get_rays(K: np.ndarray, c2w: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> Rays:
    dirs = torch.stack([(xs - K[0, 2]) / K[0, 0], -(ys - K[1, 2]) / K[1, 1], -torch.ones_like(xs)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # Normalize directions
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return Rays(rays_o, rays_d)


def pos_smoke2world(Psmoke, s2w):
    pos_scale = Psmoke  # 2.simulation to 3.target
    pos_rot = torch.sum(pos_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    pos_off = (s2w[:3, -1]).expand(pos_rot.shape)  # 3.target to 4.world
    return pos_rot + pos_off


def convert_aabb(in_min, in_max, voxel_tran):
    in_min = torch.tensor(in_min, device=voxel_tran.device).expand(3)
    in_max = torch.tensor(in_max, device=voxel_tran.device).expand(3)
    in_min = pos_smoke2world(in_min, voxel_tran)
    in_max = pos_smoke2world(in_max, voxel_tran)
    cmp = torch.less(in_min, in_max)
    in_min, in_max = torch.where(cmp, in_min, in_max), torch.where(cmp, in_max, in_min)
    return torch.cat((in_min, in_max))


def img2mse(x: torch.Tensor, y: torch.Tensor):
    return torch.mean((x - y) ** 2)


def mse2psnr(x: torch.Tensor):
    return -10. * torch.log10(x)


def to8b(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# ======================================== Velocity ========================================

def get_voxel_pts(H, W, D, s2w, n_jitter=0, r_jitter=0.8):
    """Get voxel positions."""

    i, j, k = torch.meshgrid(
        torch.linspace(0, D - 1, D),
        torch.linspace(0, H - 1, H),
        torch.linspace(0, W - 1, W))
    pts = torch.stack([(k + 0.5) / W, (j + 0.5) / H, (i + 0.5) / D], -1).to(s2w.device)
    # shape D*H*W*3, value [(x,y,z)] , range [0,1]

    jitter_r = torch.Tensor([r_jitter / W, r_jitter / H, r_jitter / D]).float().expand(pts.shape).to(s2w.device)
    for i_jitter in range(n_jitter):
        off_i = torch.rand(pts.shape, dtype=torch.float) - 0.5
        # shape D*H*W*3, value [(x,y,z)] , range [-0.5,0.5]

        pts = pts + off_i * jitter_r

    return pos_smoke2world(pts, s2w)


def get_density_flat(cur_pts, chunk=1024 * 32, network_fn: RadianceField = None, getStatic=True, **kwargs):
    input_shape = list(cur_pts.shape[0:-1])

    pts_flat = cur_pts.view(-1, cur_pts.shape[-1])
    pts_N = pts_flat.shape[0]
    # Evaluate model
    all_sigma = []
    for i in range(0, pts_N, chunk):
        pts_i = pts_flat[i:i + chunk]

        if isinstance(network_fn, HybridRadianceField):
            # kwargs["opaque"] = True
            raw_i = network_fn.static_model.query_density(pts_i[..., :3], **kwargs)
            raw_j = network_fn.dynamic_model.query_density(pts_i)
            all_sigma.append(torch.cat([raw_i, raw_j], -1))
        else:
            raw_i = network_fn.query_density(pts_i)
            all_sigma.append(raw_i)

    all_sigma = torch.cat(all_sigma, 0).view(input_shape + [-1])
    den_raw = all_sigma[..., -1:]
    returnStatic = getStatic and (all_sigma.shape[-1] > 1)
    if returnStatic:
        static_raw = all_sigma[..., :1]
        return [den_raw, static_raw]
    return [den_raw]


def get_velocity_flat(cur_pts, chunk=1024 * 32, vel_model=None):
    pts_N = cur_pts.shape[0]
    world_v = []
    for i in range(0, pts_N, chunk):
        input_i = cur_pts[i:i + chunk]
        vel_i = vel_model(input_i)
        world_v.append(vel_i)
    world_v = torch.cat(world_v, 0)
    return world_v


def vel_world2smoke(Vworld, w2s, st_factor):
    vel_rot = Vworld[..., None, :] * (w2s[:3, :3])
    vel_rot = torch.sum(vel_rot, -1)  # 4.world to 3.target
    vel_scale = vel_rot * st_factor  # 3.target to 2.simulation
    return vel_scale


def vel_smoke2world(Vsmoke, s2w, st_factor):
    vel_scale = Vsmoke / st_factor  # 2.simulation to 3.target
    vel_rot = torch.sum(vel_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return vel_rot


def pos_world2smoke(Pworld, w2s):
    # pos_rot = torch.sum(Pworld[..., None, :] * (w2s[:3,:3]), -1) # 4.world to 3.target
    pos_rot = (w2s[:3, :3] @ Pworld[..., :, None]).squeeze()
    pos_off = (w2s[:3, -1]).expand(pos_rot.shape)  # 4.world to 3.target
    new_pose = pos_rot + pos_off
    return new_pose


def off_smoke2world(Offsmoke, s2w):
    off_scale = Offsmoke  # 2.simulation to 3.target
    off_rot = torch.sum(off_scale[..., None, :] * (s2w[:3, :3]), -1)  # 3.target to 4.world
    return off_rot


def den_scalar2rgb(den, scale: float | None = 160.0, is3D=False, logv=False, mix=True):
    # den: a np.float32 array, in shape of (?=b,) d,h,w,1 for 3D and (?=b,)h,w,1 for 2D
    # scale: scale content to 0~255, something between 100-255 is usually good.
    #        content will be normalized if scale is None
    # logv: visualize value with log
    # mix: use averaged value as a volumetric visualization if True, else show middle slice

    ori_shape = list(den.shape)
    if ori_shape[-1] != 1:
        ori_shape.append(1)
        den = np.reshape(den, ori_shape)

    if is3D:
        new_range = list(range(len(ori_shape)))
        z_new_range = new_range[:]
        z_new_range[-4] = new_range[-3]
        z_new_range[-3] = new_range[-4]
        # print(z_new_range)
        YZXden = np.transpose(den, z_new_range)

        if not mix:
            _yz = YZXden[..., (ori_shape[-2] - 1) // 2, :]
            _yx = YZXden[..., (ori_shape[-4] - 1) // 2, :, :]
            _zx = YZXden[..., (ori_shape[-3] - 1) // 2, :, :, :]
        else:
            _yz = np.average(YZXden, axis=-2)
            _yx = np.average(YZXden, axis=-3)
            _zx = np.average(YZXden, axis=-4)
            # print(_yx.shape, _yz.shape, _zx.shape)

        # in case resolution is not a cube, (res,res,res)
        _yxz = np.concatenate([  # yz, yx, zx
            _yx, _yz], axis=-2)  # (?=b,),h,w+zdim,1

        if ori_shape[-3] < ori_shape[-4]:
            pad_shape = list(_yxz.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-4] - ori_shape[-3]
            _pad = np.zeros(pad_shape, dtype=np.float32)
            _yxz = np.concatenate([_yxz, _pad], axis=-3)
        elif ori_shape[-3] > ori_shape[-4]:
            pad_shape = list(_zx.shape)  # (?=b,),h,w+zdim,1
            pad_shape[-3] = ori_shape[-3] - ori_shape[-4]

            _zx = np.concatenate(
                [_zx, np.zeros(pad_shape, dtype=np.float32)], axis=-3)

        midDen = np.concatenate([  # yz, yx, zx
            _yxz, _zx
        ], axis=-2)  # (?=b,),h,w*3,1
    else:
        midDen = den

    if logv:
        midDen = np.log10(midDen + 1)
    if scale is None:
        midDen = midDen / max(midDen.max(), 1e-6) * 255.0
    else:
        midDen = midDen * scale
    grey = np.clip(midDen, 0, 255)

    return grey.astype(np.uint8)[::-1]  # flip y


class VoxelTool(object):

    def __get_tri_slice(self, _xm, _ym, _zm, _n=1):
        _yz = torch.reshape(self.pts[..., _xm:_xm + _n, :], (-1, 3))
        _zx = torch.reshape(self.pts[:, _ym:_ym + _n, ...], (-1, 3))
        _xy = torch.reshape(self.pts[_zm:_zm + _n, ...], (-1, 3))

        pts_mid = torch.cat([_yz, _zx, _xy], dim=0)
        npMaskXYZ = [np.zeros([self.D, self.H, self.W, 1], dtype=np.float32) for _ in range(3)]
        npMaskXYZ[0][..., _xm:_xm + _n, :] = 1.0
        npMaskXYZ[1][:, _ym:_ym + _n, ...] = 1.0
        npMaskXYZ[2][_zm:_zm + _n, ...] = 1.0
        return pts_mid, torch.tensor(np.clip(npMaskXYZ[0] + npMaskXYZ[1] + npMaskXYZ[2], 1e-6, 3.0), device=pts_mid.device)

    def __pad_slice_to_volume(self, _slice, _n, mode=0):
        # mode: 0, x_slice, 1, y_slice, 2, z_slice
        tar_shape = [self.D, self.H, self.W]
        in_shape = tar_shape[:]
        in_shape[-1 - mode] = _n
        fron_shape = tar_shape[:]
        fron_shape[-1 - mode] = (tar_shape[-1 - mode] - _n) // 2
        back_shape = tar_shape[:]
        back_shape[-1 - mode] = (tar_shape[-1 - mode] - _n - fron_shape[-1 - mode])

        cur_slice = _slice.view(in_shape + [-1])
        front_0 = torch.zeros(fron_shape + [cur_slice.shape[-1]], device=_slice.device)
        back_0 = torch.zeros(back_shape + [cur_slice.shape[-1]], device=_slice.device)

        volume = torch.cat([front_0, cur_slice, back_0], dim=-2 - mode)
        return volume

    def __init__(self, voxel_tran: torch.Tensor, voxel_tran_inv: torch.Tensor, scene_scale: np.ndarray,
                 x: int, middle_view: bool = True):
        assert scene_scale[0] == 1.0  # normalized by x /= x[0]
        scene_size = (scene_scale * x).round().astype(int)
        W, H, D = scene_size
        self.s_s2w = voxel_tran
        self.s_w2s = voxel_tran_inv
        self.D = D
        self.H = H
        self.W = W
        self.pts = get_voxel_pts(H, W, D, self.s_s2w)
        self.pts_mid = None
        self.mask_xyz = None
        self.middle_view = middle_view
        if middle_view is not None:
            _n = 1 if middle_view else 3
            _xm, _ym, _zm = (W - _n) // 2, (H - _n) // 2, (D - _n) // 2
            self.pts_mid, self.mask_xyz = self.__get_tri_slice(_xm, _ym, _zm, _n)

    def voxel_size(self) -> tuple[int, int, int]:
        return self.W, self.H, self.D

    def get_voxel_density_list(self, t=None, chunk=1024 * 32, network_fn=None, middle_slice=False, **kwargs):
        D, H, W = self.D, self.H, self.W
        # middle_slice, only for fast visualization of the middle slice
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        if t is not None:
            pts_flat = attach_time(pts_flat, t)

        den_list = get_density_flat(pts_flat, chunk, network_fn, **kwargs)

        return_list = []
        for den_raw in den_list:
            if middle_slice:
                # only for fast visualization of the middle slice
                _n = 1 if self.middle_view else 3
                _yzV, _zxV, _xyV = torch.split(den_raw, [D * H * _n, D * W * _n, H * W * _n], dim=0)
                mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) + self.__pad_slice_to_volume(_xyV, _n, 2)
                return_list.append(mixV / self.mask_xyz)
            else:
                return_list.append(den_raw.view(D, H, W, 1))
        return return_list

    def get_voxel_velocity(self, deltaT, t, chunk=1024 * 32,
                           vel_model=None, middle_slice=False, ref_den_list=None):
        # middle_slice, only for fast visualization of the middle slice
        D, H, W = self.D, self.H, self.W
        pts_flat = self.pts_mid if middle_slice else self.pts.view(-1, 3)
        if t is not None:
            pts_flat = attach_time(pts_flat, t)

        world_v = get_velocity_flat(pts_flat, chunk, vel_model)
        reso_scale = torch.tensor([self.W * deltaT, self.H * deltaT, self.D * deltaT], device=pts_flat.device)
        target_v = vel_world2smoke(world_v, self.s_w2s, reso_scale)

        if middle_slice:
            _n = 1 if self.middle_view else 3
            _yzV, _zxV, _xyV = torch.split(target_v, [D * H * _n, D * W * _n, H * W * _n], dim=0)
            mixV = self.__pad_slice_to_volume(_yzV, _n, 0) + self.__pad_slice_to_volume(_zxV, _n, 1) + self.__pad_slice_to_volume(_xyV, _n, 2)
            target_v = mixV / self.mask_xyz
        else:
            target_v = target_v.view(D, H, W, 3)

        if ref_den_list is not None:
            target_v = target_v - target_v * torch.less(ref_den_list, 0.1) * 0.5

        return target_v

    def save_voxel_den_npz(self, den_path, t, network_fn=None, chunk=1024 * 32, save_npz=True, save_jpg=False, jpg_mix=True,
                           noStatic=False, **kwargs):
        voxel_den_list = self.get_voxel_density_list(t, chunk, network_fn, middle_slice=not (jpg_mix or save_npz), **kwargs)
        head_tail = os.path.split(den_path)
        namepre = ["", "static_"]
        for voxel_den, npre in zip(voxel_den_list, namepre):
            voxel_den = voxel_den.detach().cpu().numpy()
            if save_jpg:
                jpg_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".jpg")
                imageio.imwrite(jpg_path, den_scalar2rgb(voxel_den, scale=None, is3D=True, logv=False, mix=jpg_mix).squeeze())
            if save_npz:
                # to save some space
                npz_path = os.path.join(head_tail[0], npre + os.path.splitext(head_tail[1])[0] + ".npz")
                voxel_den = np.float16(voxel_den)
                np.savez_compressed(npz_path, vel=voxel_den)
            if noStatic:
                break

# from FFJORD github code
def _get_minibatch_jacobian(y, x):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.
    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(
            y[:, j],
            x,
            torch.ones_like(y[:, j], device=y.get_device()),
            retain_graph=True,
            create_graph=True,
        )[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac

def get_density_and_derivatives(cur_pts, chunk=1024*32, network_fn=None, **kwargs):
    _den = get_density_flat(cur_pts, chunk, network_fn, False, **kwargs)[0]
    # requires 1 backward passes
    # The minibatch Jacobian matrix of shape (N, D_y=1, D_x=4)
    jac = _get_minibatch_jacobian(_den, cur_pts)
    jac = torch.where(torch.isnan(jac), 0.0, jac)   # fix for s-density in neus
    # assert not torch.any(torch.isnan(jac))
    _d_x, _d_y, _d_z, _d_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,1)
    return _den, _d_x, _d_y, _d_z, _d_t


def get_velocity_and_derivatives(cur_pts, chunk=1024*32, vel_model=None):
    _vel = get_velocity_flat(cur_pts, chunk, vel_model)
    # requires 3 backward passes
    # The minibatch Jacobian matrix of shape (N, D_y=3, D_x=4)
    jac = _get_minibatch_jacobian(_vel, cur_pts)
    _u_x, _u_y, _u_z, _u_t = [torch.squeeze(_, -1) for _ in jac.split(1, dim=-1)] # (N,3)
    return _vel, _u_x, _u_y, _u_z, _u_t

def PDE_EQs(D_t, D_x, D_y, D_z, U, U_t=None, U_x=None, U_y=None, U_z=None):
    eqs = []
    dts = [D_t]
    dxs = [D_x]
    dys = [D_y]
    dzs = [D_z]

    if None not in [U_t, U_x, U_y, U_z]:
        dts += U_t.split(1, dim = -1) # [d_t, u_t, v_t, w_t] # (N,1)
        dxs += U_x.split(1, dim = -1) # [d_x, u_x, v_x, w_x]
        dys += U_y.split(1, dim = -1) # [d_y, u_y, v_y, w_y]
        dzs += U_z.split(1, dim = -1) # [d_z, u_z, v_z, w_z]

    u,v,w = U.split(1, dim=-1) # (N,1)
    for dt, dx, dy, dz in zip (dts, dxs, dys, dzs):
        _e = dt + (u*dx + v*dy + w*dz)
        eqs += [_e]
    # transport and nse equations:
    # e1 = d_t + (u*d_x + v*d_y + w*d_z) - PecInv*(c_xx + c_yy + c_zz)          , should = 0
    # e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - ReyInv*(u_xx + u_yy + u_zz)    , should = 0
    # e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - ReyInv*(v_xx + v_yy + v_zz)    , should = 0
    # e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - ReyInv*(w_xx + w_yy + w_zz)    , should = 0
    # e5 = u_x + v_y + w_z                                                      , should = 0
    # For simplification, we assume PecInv = 0.0, ReyInv = 0.0, pressure p = (0,0,0)

    if None not in [U_t, U_x, U_y, U_z]:
        # eqs += [ u_x + v_y + w_z ]
        eqs += [ dxs[1] + dys[2] + dzs[3] ]

    if True: # scale regularization
        eqs += [ (u*u + v*v + w*w)* 1e-1]

    return eqs

# ======================================== Velocity ========================================

def save_model(path: str, global_step: int,
               model: RadianceField, prop_model: RadianceField | None, optimizer: torch.optim.Optimizer,
               vel_model=None, vel_optimizer=None):
    save_dic = {
        'global_step': global_step,
        'network_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if prop_model is not None:
        save_dic['network_prop_state_dict'] = prop_model.state_dict()

    if vel_model is not None:
        save_dic['network_vel_state_dict'] = vel_model.state_dict()
        save_dic['vel_optimizer_state_dict'] = vel_optimizer.state_dict()

    torch.save(save_dic, path)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    ### ==================== LOAD CONFIG ==================== ###
    # scene = "sphere_neus"
    # args_dict = np.load(f"args/args_{scene}.npy", allow_pickle=True).item()
    # args = argparse.Namespace(**args_dict)
    ### ==================== LOAD CONFIG ==================== ###

    ### ==================== TEMP ARGS ==================== ###
    fading_layers = 50000
    net_model = "siren"
    s_model = "neus"
    netdepth, netwidth = 8, 256
    use_viewdirs = False
    omega = 30.0
    use_first_omega = False
    multires = 0
    multires_views = 0
    use_color_t = False
    N_importance = 64
    lrate = 0.0005
    vel_delay = 20000
    n_rand = 1024
    tempo_fading_in = 2000
    datadir = "./data/pinf/Sphere"
    half_res = "normal"
    testskip = 20
    vgg_strides = 4
    precrop_iters = 500
    precrop_frac = 0.5
    N_samples = 32
    perturb = 1.0
    bbox_min = 0.0
    bbox_max = 1.0
    chunk = 4096
    vggW = 0.003
    ghostW = 0.003
    ghost_scale = 9.0
    overlayW = 0.002
    eikonal = 0.01
    devW = 0.0
    lrate_decay = 500
    vol_output_W = 128
    vel_no_slip = False
    nseW = 0.001
    neumann = 1.0
    ### ==================== TEMP ARGS ==================== ###

    model = HybridRadianceField(
        static_model=NeuS(D=netdepth, W=netwidth, input_ch=3, multires=multires, multires_views=multires_views, use_color_t=use_color_t, output_s_density=False).to(device),
        dynamic_model=SIREN_NeRFt(D=netdepth, W=netwidth, input_ch=4, use_viewdirs=use_viewdirs, first_omega_0=omega, unique_first=use_first_omega, fading_fin_step=fading_layers).to(device)
    )
    vel_model = SIREN_vel(fading_fin_step=fading_layers).to(device)
    prop_model = SIREN_NeRFt(D=netdepth, W=netwidth, input_ch=4, use_viewdirs=use_viewdirs, first_omega_0=omega, unique_first=use_first_omega, fading_fin_step=fading_layers).to(device)
    grad_vars = list(model.static_model.parameters()) + list(model.dynamic_model.parameters()) + list(prop_model.parameters())
    vel_grad_vars = list(vel_model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=lrate, betas=(0.9, 0.999))
    vel_optimizer = torch.optim.Adam(params=vel_grad_vars, lr=lrate, betas=(0.9, 0.999))

    ### ==================== PREPARE FOR ITERATION ==================== ###
    start = 0
    global_step = start
    n_iters = 100
    velInStep = vel_delay
    ### ==================== PREPARE FOR ITERATION ==================== ###
    vgg_tool = VGGLossTool(device)
    pinf_data = PINFFrameData(datadir, half_res=half_res, normalize_time=True)
    train_data = PINFDataset(pinf_data)
    test_data = PINFTestDataset(pinf_data, skip=testskip, video_id=0)
    t_info = pinf_data.t_info
    bkg_color = torch.tensor(pinf_data.bkg_color, device=device)
    in_min = [bbox_min]
    in_max = [bbox_max]
    voxel_tran = pinf_data.voxel_tran
    voxel_tran[:3, :3] *= pinf_data.voxel_scale
    voxel_tran = torch.tensor(voxel_tran, device=device)
    voxel_tran_inv = torch.inverse(voxel_tran)
    scene_scale = pinf_data.voxel_scale / pinf_data.voxel_scale[0]
    aabb = convert_aabb(in_min, in_max, voxel_tran)

    min_ratio = float(64 + 4 * 2) / min(scene_scale[0], scene_scale[1], scene_scale[2])
    train_x = max(vol_output_W, int(min_ratio * scene_scale[0] + 0.5))
    training_voxel = VoxelTool(voxel_tran, voxel_tran_inv, scene_scale, train_x)
    training_pts = torch.reshape(training_voxel.pts, (-1, 3))
    voxel_writer = training_voxel

    split_nse_wei = [2.0, 1e-3, 1e-3, 1e-3, 5e-3, 5e-3]

    renderer = PINFRenderer(
        model=model,
        prop_model=prop_model,
        n_samples=N_samples,
        n_importance=N_importance,
        near=pinf_data.near,
        far=pinf_data.far,
        perturb=perturb > 0,
        vel_model=vel_model,
        aabb=aabb,
    )

    model_fading_update(model, prop_model, vel_model, start, velInStep)
    for i in tqdm.trange(n_iters):
        model_fading_update(model, prop_model, vel_model, global_step, velInStep)
        tempo_fading = fade_in_weight(global_step, 0, tempo_fading_in)
        vel_fading = fade_in_weight(global_step, velInStep, 10000)
        warp_fading = fade_in_weight(global_step, velInStep + 10000, 20000)
        vgg_fading = [fade_in_weight(global_step, (vgg_i - 1) * 10000, 10000) for vgg_i in range(len(vgg_tool.layer_list), 0, -1)]
        ghost_fading = fade_in_weight(global_step, 2000, 20000)

        # Random from one frame
        video, frame_i = train_data.get_video_and_frame(np.random.randint(len(train_data)))
        target = torch.tensor(video.frames[frame_i], device=device)
        K = video.intrinsics()
        H, W = target.shape[:2]
        pose = torch.tensor(video.c2w(frame_i), device=device)
        time_locate = t_info[-1] * frame_i
        background = bkg_color

        trainVel = (global_step >= velInStep) and (i % 10 == 0)
        if trainVel:
            # take a mini_batch 32*32*32
            train_x, train_y, train_z = training_voxel.voxel_size()
            train_random = np.random.choice(train_z * train_y * train_x, 32 * 32 * 32)
            training_samples = training_pts[train_random]

            training_samples = training_samples.view(-1, 3)
            training_t = torch.ones([training_samples.shape[0], 1], device=device) * time_locate
            training_samples = torch.cat([training_samples, training_t], dim=-1)

            #####  core velocity optimization loop  #####
            # allows to take derivative w.r.t. training_samples
            training_samples = training_samples.detach().requires_grad_(True)
            _vel, _u_x, _u_y, _u_z, _u_t = get_velocity_and_derivatives(training_samples, chunk=chunk, vel_model=vel_model)
            if vel_no_slip:
                smoke_model = model
            else:
                smoke_model = model.dynamic_model if isinstance(model, HybridRadianceField) else model
            _den, _d_x, _d_y, _d_z, _d_t = get_density_and_derivatives(
                training_samples, chunk=chunk,
                network_fn=smoke_model,
                opaque=True,  # for neus
            )

            vel_optimizer.zero_grad()
            split_nse = PDE_EQs(
                _d_t.detach(), _d_x.detach(), _d_y.detach(), _d_z.detach(),
                _vel, _u_t, _u_x, _u_y, _u_z)
            nse_errors = [torch.mean(torch.square(x)) for x in split_nse]
            nseloss_fine = 0.0
            for ei, wi in zip(nse_errors, split_nse_wei):
                nseloss_fine = ei * wi + nseloss_fine
            vel_loss = nseloss_fine * nseW * vel_fading

            # Neumann loss
            if isinstance(model, HybridRadianceField) and isinstance(model.static_model, SDFRadianceField):
                sdf_model = model.static_model
                with torch.no_grad():
                    if isinstance(sdf_model, NeuS):
                        sdf, gradient = sdf_model.forward_with_gradient(training_samples[..., :3])
                        sdf = sdf[..., :1]
                    else:
                        sdf = sdf_model.sdf(training_samples[..., :3])
                        gradient = sdf_model.gradient(training_samples[..., :3])
                sdf, gradient = sdf.detach(), gradient.detach()
                neumann_loss = sdf_model.opaque_density(sdf).detach() * torch.nn.functional.relu(-torch.sum(_vel * gradient, dim=-1, keepdim=True))
                neumann_loss = torch.mean(neumann_loss)
                if neumann > 0.0:
                    vel_loss = vel_loss + neumann_loss * neumann
                del sdf, gradient
                neumann_loss = neumann_loss.detach()
                # writer.add_scalar('Neumann loss', neumann_loss, i)

            vel_loss.backward()
            vel_optimizer.step()

            # cleanup
            del _vel, _u_x, _u_y, _u_z, _u_t, _den, _d_x, _d_y, _d_z, _d_t, split_nse
            nse_errors = tuple(x.item() for x in nse_errors)
            nseloss_fine = nseloss_fine.item()
            vel_loss = vel_loss.item()

        trainVGG = (i % 4 == 0)
        if trainVGG:
            coords_crop, dw = vgg_sample(vgg_strides, n_rand, target, bkg_color, steps=i)
            coords_crop = torch.reshape(coords_crop, [-1, 2])
            ys, xs = coords_crop[:, 0], coords_crop[:, 1]  # vgg_sample using ij, convert to xy
        else:
            if i < precrop_iters:
                dH = int(H // 2 * precrop_frac)
                dW = int(W // 2 * precrop_frac)
                xs, ys = torch.meshgrid(
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW),
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                    indexing='xy'
                )
                selected = np.random.choice(4 * dH * dW, size=[n_rand], replace=False)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {precrop_iters}")
            else:
                xs, ys = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='xy')
                selected = np.random.choice(H * W, size=[n_rand], replace=False)
            xs = torch.flatten(xs)[selected].to(device)
            ys = torch.flatten(ys)[selected].to(device)

        rays = get_rays(K, pose, xs, ys)  # (n_rand, 3), (n_rand, 3)
        rays = rays.foreach(lambda t: t.to(device))
        target_s = target[ys.long(), xs.long()]  # (n_rand, 3)

        if global_step >= velInStep:
            renderer.warp_fading_dt = warp_fading * t_info[-1]

        #####  core radiance optimization loop  #####
        output = renderer.render(
            rays.origins, rays.viewdirs, chunk=chunk,
            ret_raw=True,
            timestep=time_locate,
            background=background)
        rgb, _, acc, extras = output.as_tuple()
        out0: NeRFOutputs | None = extras.get('coarse')

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        if 'static' in extras and tempo_fading < 1.0 - 1e-8:
            img_loss = img_loss * tempo_fading + img2mse(extras['static'].rgb, target_s) * (1.0 - tempo_fading)
            # rgb = rgb * tempo_fading + extras['rgbh1'] * (1.0-tempo_fading)
        loss = img_loss
        psnr = mse2psnr(img_loss.detach())

        if out0 is not None:
            img_loss0 = img2mse(out0.rgb, target_s)
            if 'static' in out0.extras and tempo_fading < 1.0 - 1e-8:
                img_loss0 = img_loss0 * tempo_fading + img2mse(out0.extras['static'].rgb, target_s) * (1.0 - tempo_fading)
            loss = loss + img_loss0

        if trainVGG:
            vgg_loss_func = vgg_tool.compute_cos_loss
            vgg_tar = torch.reshape(target_s, [dw, dw, 3])
            vgg_img = torch.reshape(rgb, [dw, dw, 3])
            vgg_loss = vgg_loss_func(vgg_img, vgg_tar)
            w_vgg = vggW / float(len(vgg_loss))
            vgg_loss_sum = 0
            for _w, _wf in zip(vgg_loss, vgg_fading):
                if _wf > 1e-8:
                    vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum

            if out0 is not None:
                vgg_img0 = torch.reshape(out0.rgb, [dw, dw, 3])
                vgg_loss0 = vgg_loss_func(vgg_img0, vgg_tar)
                for _w, _wf in zip(vgg_loss0, vgg_fading):
                    if _wf > 1e-8:
                        vgg_loss_sum = _w * _wf * w_vgg + vgg_loss_sum
            loss += vgg_loss_sum

        if (ghostW > 0.0) and background is not None:
            w_ghost = ghost_fading * ghostW
            if w_ghost > 1e-8:
                ghost_loss = ghost_loss_func(output, background, ghost_scale)
                if 'static' in extras:  # static part
                    ghost_loss += 0.1 * ghost_loss_func(extras['static'], background, ghost_scale)
                    if 'dynamic' in extras:  # dynamic part
                        ghost_loss += 0.1 * ghost_loss_func(extras['dynamic'], extras['static'].rgb, ghost_scale)

                if out0 is not None:
                    ghost_loss0 = ghost_loss_func(out0, background, ghost_scale)
                    if 'static' in out0.extras:  # static part
                        # ghost_loss0 += 0.1*ghost_loss_func(extras['rgbh10'], static_back, extras['acch10'], den_penalty=0.0)
                        if 'dynamic' in out0.extras:  # dynamic part
                            ghost_loss += 0.1 * ghost_loss_func(out0.extras['dynamic'], out0.extras['static'].rgb,
                                                                ghost_scale)

                    ghost_loss += ghost_loss0

                loss += ghost_loss * w_ghost

        w_overlay = overlayW * ghost_fading  # with fading
        if 'static' in extras and w_overlay > 0:
            # density should be either from smoke or from static, not mixed.
            smoke_den = extras['sigma_d']
            if 'sdf' in extras:
                inv_s = extras['inv_s'].detach()  # as constant
                static_den = inv_s * torch.sigmoid(-inv_s * extras['sdf']) / 2  # opaque_density
            else:
                static_den = extras['sigma_s']
            overlay_loss = (smoke_den * static_den) / (torch.square(smoke_den) + torch.square(static_den) + 1e-8)
            overlay_loss = torch.mean(overlay_loss)
            loss += overlay_loss * w_overlay

        eikonal_loss = extras.get('eikonal_loss')
        eikonal_weight = eikonal
        if eikonal_loss is not None and eikonal_weight > 0:
            # eikonal_weight = np.clip(i / 10000, 0.0, 1.0) * 0.001
            loss += eikonal_loss * eikonal_weight
            # writer.add_scalar('eikonal_loss', eikonal_loss, i)
            if i > 20000 and extras['inv_s'] < 100.0 and devW > 0:
                loss += devW / extras['inv_s']

        loss.backward()
        optimizer.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        if trainVel and vel_optimizer is not None:
            for param_group in vel_optimizer.param_groups:
                param_group['lr'] = new_lrate

        print(f"loss: {loss.item():.4f}, img_loss: {img_loss.item():.4f}, psnr: {psnr.item():.2f}")

        i_weights = 2000
        expdir = os.path.join('logs', 'sphere_neux')
        os.makedirs(expdir, exist_ok=True)
        if (i in (100, 10000, 20000, 40000) or i % i_weights == 0) and i > start + 1:
            path = os.path.join(expdir, '{:06d}.tar'.format(i))
            save_model(path, global_step, model, prop_model, optimizer, vel_model, vel_optimizer)
            print('Saved checkpoints at', path)

        global_step += 1
