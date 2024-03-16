import torch
from torch import nn 
from torch._prims_common import Tensor
import torch.nn.functional as F



class BitLinear(nn.Linear):
    """BitLinear is a linear layer with bit-wise operations
    It is a drop-in replacement for the standard linear layer
    that is less computationally expensive but competitively accurate.
    its based on the paper.

        https://arxiv.org/pdf/2310.11453.pdf

    BitLinear : input -> Quantization(absmax) -> BitWeight product -> Dequantization + Absmax quants
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bit: int = 1,
        device = None,
        dtype = None
    ):
        super(BitLinear, self).__init__(in_features, out_features, bias, device, dtype)
        self.bit = bit
        self.ln = nn.LayerNorm(in_features)
        self.eps = 1e-05 # the eps usually used is 1e-05
        #self.alpha = torch.sum(self.weight) / (self.weight.size(0) * self.weight.size(0))
        self.alpha = torch.mean(self.weight)
        self.beta = self.weight.abs().mean()
        self.binarized_weight = self._ste(self.weight - self.alpha)
        self.Qb = 2 ** (bit - 1) 

    def forward(self, input):
            # normalize, quantize, transform, dequantize
            x = self.ln(input)
            self.gamma = torch.norm(x, p=torch.inf) # torch.max() will do too I guess??
            x_hat = self._quantize(x)
            y = self._dequantize(F.linear(x_hat, self.binarized_weight, self.bias))
            return y

    def _dequantize(self, x):
        return x * self.beta * self.gamma / self.Qb

    def _quantize(self, x):
        v = x * (self.Qb / self.gamma)
        w = -self.Qb + self.eps
        x = self.Qb - self.eps
        return torch.clip(v, w, x) 

    def _ste(self, x: Tensor):
        """ Straight through estimator (STE) function

        sign(x) -> -1 if x >= 0 else 1

        """
        return (torch.sign(x) - x).detach() + x


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, bit={}'.format(
            self.in_features,
            self.out_features,
            self.bias,
            self.bit
        )

if __name__ == "__main__":
    x = BitLinear(5,5, bias=False)
    print(x)
    e = x(torch.rand((5,5)))
    print(e)
