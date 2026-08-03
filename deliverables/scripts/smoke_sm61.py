import torch

print("torch:", torch.__version__)
print("cuda(build):", torch.version.cuda)
print("arch_list:", torch.cuda.get_arch_list())
print("capability:", torch.cuda.get_device_capability())
print("device:", torch.cuda.get_device_name(0))
print("bf16_supported:", torch.cuda.is_bf16_supported())

# Real kernel execution. A wrong wheel fails here with "no kernel image".
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
print("matmul ok:", (a @ b).sum().item())

# Confirm a cuDNN convolution path separately.
import torch.nn as nn

conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
x = torch.randn(1, 3, 512, 512, device="cuda")
print("conv ok:", conv(x).shape)

# SDPA backend availability is a key performance variable on Pascal.
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(1, 8, 1024, 64, device="cuda")
for backend in [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]:
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        print(f"SDPA {backend}: OK")
    except Exception as e:
        print(f"SDPA {backend}: FAIL ({type(e).__name__})")
