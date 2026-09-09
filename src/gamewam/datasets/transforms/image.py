import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as transforms_F


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x

class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class Normalize(nn.Module):
    """Convert to tensor if needed and normalize by mean/std."""

    def __init__(self, args):
        super().__init__()
        assert args is not None, "Please specify args"
        self.args = args

    def forward(self, video):
        mean = self.args["mean"]
        std = self.args["std"]

        if isinstance(video, torch.Tensor):
            data = video.to(dtype=torch.float32)
            if video.dtype == torch.uint8:
                data = data / 255.0
            data = data.to(dtype=torch.get_default_dtype())
        else:
            data = transforms_F.to_tensor(video)  # division by 255 is applied in to_tensor()

        return transforms_F.normalize(tensor=data, mean=mean, std=std)
