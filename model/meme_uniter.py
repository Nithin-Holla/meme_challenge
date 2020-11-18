from torch import nn

from model.model import UniterModel


class MemeUniter(nn.Module):

    def __init__(self,
                 uniter_model: UniterModel,
                 hidden_size: int,
                 n_classes: int):
        super().__init__()
        self.uniter_model = uniter_model
        self.n_classes = n_classes
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, **kwargs):
        out = self.uniter_model(**kwargs)
        out = self.uniter_model.pooler(out)
        out = self.linear(out)
        return out
