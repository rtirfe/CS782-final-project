import torch

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv import SAGEConv, GATConv
from torch.nn import ModuleList, Embedding
from torch.nn.modules.loss import _Loss
from torch.nn.functional import logsigmoid

class MyGNN(torch.nn.Module):

    def __init__(self, 
                 embedding_dim, 
                 num_layers,
                 **kwargs):
        super(MyGNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        convs = []
        convs.append(GATConv(in_channels=11, out_channels=embedding_dim, **kwargs))
        for _ in range(num_layers-1):
            convs.append(SAGEConv(in_channels=embedding_dim, out_channels=embedding_dim, **kwargs))
        self.convs = ModuleList(convs)

        self.reset_parameters()

        alpha = 1. / (num_layers + 1)
        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))

        self.register_buffer('alpha', alpha)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, x: Tensor, edge_index: Adj) -> Tensor:

        weights = self.alpha.softmax(dim=-1)
        x = self.convs[0](x, edge_index)
        out = x * weights[0]

        for i in range(1, self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * weights[i + 1]

        return out

    def forward(self, x: Tensor, edge_index: Adj,
                edge_label_index: OptTensor = None) -> Tensor:
        if edge_label_index is None:
            if isinstance(edge_index, SparseTensor):
                edge_label_index = torch.stack(edge_index.coo()[:2], dim=0)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(x, edge_index)

        return self.predict_link_embedding(out, edge_label_index)
    
    def predict_link_embedding(self, embed: Adj, edge_label_index: Adj) -> Tensor:
        embed_src = embed[edge_label_index[0]]
        embed_dst = embed[edge_label_index[1]]
        return (embed_src * embed_dst).sum(dim=-1)

    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            lambda_reg: float = 0, **kwargs) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss."""
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, None)



class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = logsigmoid(positives - negatives).sum()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs
