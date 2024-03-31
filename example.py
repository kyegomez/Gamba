import torch
from gamba_torch.main import Gamba


# Forward pass of the GambaDecoder module.
x = torch.randn(1, 1000, 512)

# Model
model = Gamba(dim=512, d_state=512, d_conv=512, n=16384, depth=3)

# Out
out = model(x)
print(out)
