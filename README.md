# PINFS Validation Only

# Environment Setup

```shell
git clone --recursive https://github.com/zduan3/pinfs_code.git
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
python -m pip install imageio[ffmpeg] configargparse tqdm opencv-python tensorboardX pymcubes trimesh lpips
"C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build/vcvars64.bat"
set TCNN_CUDA_ARCHITECTURES=86
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
