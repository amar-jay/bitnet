import torch
from torch import nn 
from torch._prims_common import Tensor
from torch.functional import F



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
        self.eps = 1e-8
        self.ln = nn.LayerNorm(in_features)
        #self.alpha = torch.sum(self.weight) / (self.weight.size(0) * self.weight.size(0))
        self.alpha = torch.mean(self.weight)
        self.beta = self.weight.abs().mean()
        self.binarized_weight = self._ste(self.weight - self.alpha)
        self.Qb = 2 ** (bit - 1) 

    def forward(self, input):
            x = self.binarized_weight @ self._quantize(input)
            x = torch.var(x)
            return torch.sign(input) * torch.sign(self.weight) * (2 ** (self.bit - 1))

    def _quantize(self, x):
        gamma = torch.norm(x, p=torch.inf) # torch.max() will do too I guess??
        v = x * (self.Qb / gamma)
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
            self.in_features, self.out_features, self.bias is not None, self.bit
        )

