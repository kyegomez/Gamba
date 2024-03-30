import torch
from torch import nn, Tensor, einsum
from torch.nn import functional as F
from einops import rearrange, repeat
from zeta.nn import MambaBlock, FeedForward

class GambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        n: int = 16384
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
                
        self.mamba_block = MambaBlock(
            dim,
            1,
            d_state,
            d_conv=d_conv
        )
        
    def forward(self, img: Tensor, three_dgs: Tensor):
        pass
        
        

class GambaDecoder(nn.Module):
    """
    GambaDecoder is a class that represents a decoder module in the Gamba model, 
    consist of 16384 tokens, each with 512 dimensions, corresponding to 16384
    3D Gaussians. The Gaussian Decoder is an multi-layer perceptron (MLP) with 10 layers and 64
    hidden dimensions, which decodes the output 3D Gaussian of shape (16384, 23) for splattin

    Args:
        dim (int): The input dimension.
        mult (int, optional): The multiplier for the hidden dimension. Defaults to 4.
    """
    def __init__(
        self,
        dim: int,
        mult: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        
        self.mlp = FeedForward(
            dim,
            dim,
            mult,
            swish=True,
            post_act_ln=True,
        )
        
    def forward(self, x: Tensor):
        """
        Forward pass of the GambaDecoder module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the MLP.
        """
        return self.mlp(x)