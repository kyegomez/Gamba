import torch
from torch import nn, Tensor
from zeta.nn import MambaBlock, FeedForward


def prepend(to_prepend, base):
    return torch.cat((to_prepend, base), dim=1)


def drop(tensor, indices):
    """
    Drops elements from a tensor based on the given indices.

    Args:
        tensor (torch.Tensor): The input tensor.
        indices (list or torch.Tensor): The indices of the elements to be dropped.

    Returns:
        torch.Tensor: The tensor with the specified elements dropped.
    """
    mask = torch.ones(
        tensor.size(0),
        dtype=torch.bool,
    )
    mask[indices] = False
    return tensor[mask]


def Drop(tensor, num_tokens_to_drop: int):
    return tensor[:, num_tokens_to_drop:, :]


class GambaBlock(nn.Module):
    def __init__(
        self, dim: int, d_state: int, d_conv: int, n: int = 16384
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv

        self.mamba_block = MambaBlock(dim, 1, d_state, d_conv=d_conv)

        # self.three_dgs = nn.Parameter(torch.randn(n, self.dim))
        self.three_dgs = nn.Parameter(torch.randn(1, n, self.dim))

    def forward(self, img: Tensor, img_pose_token: Tensor):
        b, s, d = img.shape

        # Linear the camera pose token and image tokens
        img = nn.Linear(d, self.dim)(img + img_pose_token)
        print(img.shape)

        # Prepend with three_dgs
        img = prepend(self.three_dgs, img)
        print(img.shape)

        # MambaBlock
        mambaed = self.mamba_block(img)

        # Drop the img and img_pose_token
        num_tokens_to_drop = img_pose_token.size(1) + img.size(1)
        mambaed = Drop(mambaed, num_tokens_to_drop)
        print(mambaed.shape)

        return mambaed


# x = torch.randn(1, 64, 512)

# block = GambaBlock(dim=512, d_state=16, d_conv=4, n=16384)

# out = block(x, x)

# print(out)


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
        mult: int = 4,
        num_gaussians: int = 3,
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

        # output layers for each Gaussian Attribute
        self.position_layer = nn.Linear(dim, 3)
        self.opacity_layer = nn.Linear(dim, num_gaussians)
        self.color_layer = nn.Linear(dim, 12)

        # Decoder layer
        self.decoder_layer = nn.Linear(dim, self.dim)

    def forward(self, x: Tensor):
        """
        Forward pass of the GambaDecoder module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the MLP.
        """
        features = self.mlp(x)
        # print(f"Features: {features.shape}")

        # Position, opacity, color
        position = self.position_layer(features)
        opacity = self.opacity_layer(features)
        color = self.color_layer(features)
        print(
            f"Position: {position.shape}, Opacity: {opacity.shape},"
            f" Color: {color.shape}"
        )

        # Normalize the position to be within [-1, 1] using tanh
        position = torch.tanh(position)
        print(position)

        return {
            "position": position,
            "opacity": opacity,
            "color": color,
        }


# x = torch.randn(1, 1000, 512)

# decoder = GambaDecoder(512)

# out = decoder(x)
# print(out)


class Gamba(nn.Module):
    """
    Gamba module for image processing.

    Args:
        dim (int): Dimension of the input image.
        d_state (int): Dimension of the state.
        d_conv (int): Dimension of the convolutional layer.
        n (int, optional): Number of elements in the input image. Defaults to 16384.
        depth (int, optional): Number of Gamba blocks. Defaults to 3.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): Dimension of the input image.
        d_state (int): Dimension of the state.
        d_conv (int): Dimension of the convolutional layer.
        layers (nn.ModuleList): List of Gamba blocks.
        decoder (GambaDecoder): Gamba decoder.

    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        n: int = 16384,
        depth: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv

        # Gamba Layers
        self.layers = nn.ModuleList(
            [
                GambaBlock(
                    dim=dim, d_state=d_state, d_conv=d_conv, n=n
                )
                for _ in range(depth)
            ]
        )

        # Decoder
        self.decoder = GambaDecoder(dim)

    def forward(self, img: Tensor, *args):
        """
        Forward pass of the Gamba module.

        Args:
            img (Tensor): Input image tensor.
            *args: Variable length argument list.

        Returns:
            Tensor: Output tensor after passing through the Gamba module.

        """
        img = nn.LayerNorm(self.dim)(img)

        for layer in self.layers:
            img = layer(img, img, *args)

        return self.decoder(img)
