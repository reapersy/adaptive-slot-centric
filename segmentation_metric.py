import torch
import torch.nn.functional as F
import ipdb
st = ipdb.set_trace


def adjusted_rand_index(true_mask, pred_mask, name='ari_score'):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this function can ignore points in an image
    corresponding to the background (i.e. not to an object).
    Args:
    true_mask: `Tensor` of shape [batch_size, n_points, n_true_groups].
      The true cluster assignment encoded as one-hot.
    pred_mask: `Tensor` of shape [batch_size, n_points, n_pred_groups].
      The predicted cluster assignment encoded as categorical probabilities.
      This function works on the argmax over axis 2.
    name: str. Name of this operation (defaults to "ari_score").
    Returns:
    ARI scores as a tf.float32 `Tensor` of shape [batch_size].
    Raises:
    ValueError: if n_points <= n_true_groups and n_points <= n_pred_groups.
      We've chosen not to handle the special cases that can occur when you have
      one cluster per datapoint (which would be unusual).
    References:
    Lawrence Hubert, Phipps Arabie. 1985. "Comparing partitions"
      https://link.springer.com/article/10.1007/BF01908075
    Wikipedia
      https://en.wikipedia.org/wiki/Rand_index
    Scikit Learn
      http://scikit-learn.org/stable/modules/generated/\
      sklearn.metrics.adjusted_rand_score.html
    """
    _, n_points, n_true_groups = true_mask.shape
    n_pred_groups = pred_mask.shape[-1]
    if n_points <= n_true_groups and n_points <= n_pred_groups:
      # This rules out the n_true_groups == n_pred_groups == n_points
      # corner case, and also n_true_groups == n_pred_groups == 0, since
      # that would imply n_points == 0 too.
      # The sklearn implementation has a corner-case branch which does
      # handle this. We chose not to support these cases to avoid counting
      # distinct clusters just to check if we have one cluster per datapoint.
      raise ValueError(
          "adjusted_rand_index requires n_groups < n_points. We don't handle "
          "the spec