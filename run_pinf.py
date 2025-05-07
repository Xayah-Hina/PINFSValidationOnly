import torch
import numpy as np
import argparse
import abc


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
    velInStep = vel_delay
    ### ==================== PREPARE FOR ITERATION ==================== ###
    model_fading_update(model, prop_model, vel_model, start, velInStep)
