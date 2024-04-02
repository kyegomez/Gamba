import torch  # Importing the torch library for deep learning operations
from gamba_torch.main import (
    Gamba,
)  # Importing the Gamba module from the gamba_torch package

# Forward pass of the GambaDecoder module.
x = torch.randn(
    1, 1000, 512
)  # Generating a random tensor of shape (1, 1000, 512)

# Model
model = Gamba(
    dim=512, d_state=512, d_conv=512, n=16384, depth=3
)  # Creating an instance of the Gamba class with specified parameters

# Out
out = model(
    x
)  # Performing a forward pass of the model on the input tensor x
print(out)  # Printing the output tensor
