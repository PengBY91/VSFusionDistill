"""DetDataSample -> COCO result dicts, shared by tools/test.py and the
in-training validation hook (aodet.engine.PeriodicValHook)."""

from typing import Any, Dict, List, Mapping

import numpy as np
import torch

__all__ = ["to_coco_results"]


def to_coco_results(pred_samples, cat_id_map: Mapping[int, int]
                    ) -> List[Dict[str, Any]]:
    """DetDataSample list -> COCO result dicts (xywh, score).

    ``cat_id_map`` maps the head's label index to the GT json category id
    (position in the GT json's categories list).  Samples without a
    ``coco_image_id`` in their metainfo are skipped.
    """
    results = []
    for ds in pred_samples:
        image_id = ds.metainfo.get("coco_image_id")
        if image_id is None:
            continue  # not matched to the GT json (shouldn't happen)
        pred = ds.pred_instances
        boxes = pred.bboxes.cpu().numpy() if torch.is_tensor(pred.bboxes) \
            else np.asarray(pred.bboxes)
        scores = pred.scores.cpu().numpy() if torch.is_tensor(pred.scores) \
            else np.asarray(pred.scores)
        labels = pred.labels.cpu().numpy() if torch.is_tensor(pred.labels) \
            else np.asarray(pred.labels)
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
            results.append(dict(
                image_id=int(image_id),
                category_id=cat_id_map[int(l)],
                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                score=float(s)))
    return results
