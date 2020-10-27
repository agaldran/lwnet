from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> kornia.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.)

class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = kornia.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return dice_loss(input, target, self.eps)


class TvLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(TvLoss, self).__init__()
        self.reduction = reduction

    def compute_tv(self, logits):

        preds = torch.nn.Sigmoid()(logits)

        tv_l = torch.abs(torch.sub(preds, torch.roll(preds, shifts=1, dims=-1)))
        tv_r = torch.abs(torch.sub(preds, torch.roll(preds, shifts=-1, dims=-1)))
        tv_lu = torch.abs(torch.sub(preds, torch.roll(preds, shifts=(1, 1), dims=(-2, -1))))
        tv_rd = torch.abs(torch.sub(preds, torch.roll(preds, shifts=(-1, -1), dims=(-2, -1))))

        tv_u = torch.abs(torch.sub(preds, torch.roll(preds, shifts=1, dims=-2)))
        tv_d = torch.abs(torch.sub(preds, torch.roll(preds, shifts=-1, dims=-2)))
        tv_ru = torch.abs(torch.sub(preds, torch.roll(preds, shifts=(1, -1), dims=(-2, -1))))
        tv_dl = torch.abs(torch.sub(preds, torch.roll(preds, shifts=(-1, 1), dims=(2, -1))))

        # take mean across orientations
        tv = torch.mean(torch.stack([tv_l, tv_r, tv_lu, tv_rd, tv_u, tv_d, tv_ru, tv_dl], dim=0), dim=0)
        #         tv = torch.mean(torch.stack([tv_l, tv_r, tv_u, tv_d], dim=0), dim=0)

        return tv

    def forward(self, logits, labels, **kwargs):
        # assumes logits is bs x n_classes H x W,
        #         labels is bs x H x W containing integer values in [0,...,n_classes-1]

        tv = self.compute_tv(logits)
        masked_tv = torch.mul(tv, labels)

        # masked_tv = torch.clamp(torch.div(masked_tv, torch.nn.Sigmoid()(logits) + 1e-4), 0, 1)

        perfect_logits = labels.float()
        tv = self.compute_tv(perfect_logits)
        perfect_masked_tv = torch.mul(tv, labels)
        false_positives = (perfect_masked_tv != 0)
        masked_tv[false_positives] = 0

        if self.reduction == 'mean':  # 1 value for the entire batch
            mean_per_elem_per_class = (masked_tv.sum(dim=(-2, -1)) / labels.sum(dim=(-2, -1)))
            return mean_per_elem_per_class.mean()
        elif self.reduction == 'none':  # n_classes values per element in batch
            return masked_tv
        else:
            sys.exit('not a valid reduction scheme')