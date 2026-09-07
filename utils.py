import torch
import torch.nn as nn


def flatten_batch(feature, lengths, no_cuda):
    '''
    input shape: (B, L, D)
    output shape: (N, D)
    '''
    node_feature = []
    batch_size = feature.size(0)

    for j in range(batch_size):
        node_feature.append(feature[j, :lengths[j], :])

    node_feature = torch.cat(node_feature, dim=0)

    if not no_cuda:
        node_feature = node_feature.cuda()

    return node_feature


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params:
        num: int, the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num, requires_grad=True))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            weight = torch.exp(-self.params[i])
            total_loss += weight * loss + 0.5 * self.params[i]
        return total_loss 