import torch.nn as nn
import weakref


class HebbRepInjectPModule(nn.Module):
    """
    Wraps a PCoder's pmodule so that a Hebb module can transform the PCoder rep
    before prediction is computed.

    Key trick: PCoder.forward computes:
      self.prd = self.pmodule(self.rep)
    so we inject by making pmodule(self.rep) internally compute:
      rep_boost = hebb(self.rep)
      pcoder.rep = rep_boost
      return base_pmodule(rep_boost)
    """

    def __init__(self, base_pmodule: nn.Module, hebb: nn.Module, pcoder: nn.Module):
        super().__init__()
        self.base_pmodule = base_pmodule
        self.hebb = hebb
        # Avoid nn.Module reference cycles: keep only a weakref to the parent PCoder.
        self._pcoder_ref = weakref.ref(pcoder)

    def forward(self, rep):
        rep_boost = self.hebb(rep)
        pcoder = self._pcoder_ref()
        if pcoder is not None:
            pcoder.rep = rep_boost
        return self.base_pmodule(rep_boost)


def inject_hebb_into_pcoder_rep(net: nn.Module, pcoder_index: int, hebb: nn.Module) -> nn.Module:
    pcoder = getattr(net, f"pcoder{pcoder_index}")
    pcoder.pmodule = HebbRepInjectPModule(pcoder.pmodule, hebb, pcoder)
    return net
