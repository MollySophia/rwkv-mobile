import torch
from typing import Any

class Add(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.add(x, y)
        else:
            out = x + y
        return out

class Subtract(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.sub(x, y)
        else:
            out = x - y
        return out

class Neg(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any) -> Any:
        out = torch.neg(x)
        return out

class Multiply(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: Any, y: Any) -> Any:
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            out = torch.mul(x, y)
        else:
            out = x * y
        return out

class Tanh(torch.nn.Module):
    # pylint:disable=arguments-differ
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(x)
        return out

class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = Multiply()

    def forward(self, x: torch.Tensor) -> Any:
        return self.mul(x, self.sigmoid(x))

class Exponential(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)

class Bmm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, y)

class Split(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.split(x, *args, **kwargs)

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

class Pow(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: int) -> torch.Tensor:
        return torch.pow(x, y)