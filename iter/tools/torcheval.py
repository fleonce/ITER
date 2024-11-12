import logging
from typing import Optional, Tuple, Callable

import torch
from torcheval.metrics import MulticlassF1Score
from torcheval.metrics.classification.f1_score import TF1Score
from torcheval.metrics.functional.classification.f1_score import _f1_score_compute


def _precision_recall_f1_score_compute(
    num_tp: torch.Tensor,
    num_label: torch.Tensor,
    num_prediction: torch.Tensor,
    average: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Check if all classes exist in either ``target``.
    num_label_is_zero = num_label == 0
    if num_label_is_zero.any():
        logging.warning(
            "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros."
        )

    if average in ("macro", "weighted"):
        # Ignore the class that has no samples in both ``input`` and `target`.
        mask = (~num_label_is_zero) | (num_prediction != 0)
        num_tp, num_label, num_prediction = (
            num_tp[mask],
            num_label[mask],
            num_prediction[mask],
        )

    precision = num_tp / num_prediction
    recall = num_tp / num_label
    f1 = 2 * precision * recall / (precision + recall)

    # Convert NaN to zero when f1 score is NaN. This happens when either precision or recall is NaN or when both precision and recall are zero.
    f1 = torch.nan_to_num(f1)

    if average == "micro":
        return precision, recall, f1
    elif average == "macro":
        return precision.mean(), recall.mean(), f1.mean()
    elif average == "weighted":
        weight = (num_label / num_label.sum())
        return (precision * weight).sum(), (recall * weight).sum(), (f1 * weight).sum()
    else:  # average is None
        return precision, recall, f1


class MulticlassF1Score(MulticlassF1Score):

    @torch.inference_mode()
    def compute_with_intermediates(self: TF1Score) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return the f1 score alongside intermediate values (precision and recall).
        0 is returned if no called to ``update()`` are made before ``compute()`` is called.
        """
        return _precision_recall_f1_score_compute(
            self.num_tp, self.num_label, self.num_prediction, self.average
        )

    @torch.inference_mode()
    def compute_masked(
        self: TF1Score, ignore_index: torch.Tensor | int, average: str = None
    ) -> torch.Tensor:
        return self._compute_masked(_f1_score_compute, ignore_index, average)

    @torch.inference_mode()
    def compute_masked_with_intermediates(
        self: TF1Score, ignore_index: torch.Tensor | int, average: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._compute_masked(
            _precision_recall_f1_score_compute, ignore_index, average
        )

    @torch.inference_mode()
    def _compute_masked(
        self: TF1Score,
        compute_fn: Callable,
        ignore_index: torch.Tensor | int,
        average: str = None,
    ):
        if self.average in ("micro",):
            raise ValueError(
                "Cannot compute masked F1 when initialized with average='micro'"
            )

        num_tp = self.num_tp
        num_label = self.num_label
        num_prediction = self.num_prediction

        if ignore_index is not None:
            num_tp[ignore_index] = 0
            num_label[ignore_index] = 0
            num_prediction[ignore_index] = 0

        if average in ("micro",) and self.average not in ("average",):
            num_tp = num_tp.sum()
            num_label = num_label.sum()
            num_prediction = num_prediction.sum()

        return compute_fn(num_tp, num_label, num_prediction, average or self.average)
