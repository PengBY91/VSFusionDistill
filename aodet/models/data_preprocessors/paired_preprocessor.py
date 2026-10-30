"""Data preprocessor for paired sonar–camera batches.

The paired dataset yields per-sample dicts with ``img_camera`` (degraded
student view), ``img_camera_clear`` (pre-degradation copy for the teacher,
present only when distillation is on), ``img_sonar`` and two DetDataSample
sets (``data_samples_camera`` / ``data_samples_sonar``).  This preprocessor
stacks each image group into its own tensor and applies the *same*
normalization/channel conversion to every stream (all branches share one
YOLOX convention), mirroring mmengine's ImgDataPreprocessor per-stream
logic.

D∞-drop mode: the dataset may omit camera images entirely ("drop" rather
than "zero"); then ``inputs['camera']`` is None and the detector skips the
camera stream for that batch.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.utils import samplelist_boxtype2tensor
from mmdet.registry import MODELS as DET_MODELS
from mmdet.structures import DetDataSample
from mmengine.model.utils import stack_batch

__all__ = ["PairedDataPreprocessor"]


@DET_MODELS.register_module()
class PairedDataPreprocessor(DetDataPreprocessor):
    """Stacks {img_camera, img_camera_clear, img_sonar} separately.

    Config fields identical to DetDataPreprocessor; per-stream processing
    (channel conversion, float cast, normalization, pad-to-divisor) follows
    the ImgDataPreprocessor semantics.
    """

    def _process_stream(self, imgs) -> torch.Tensor:
        outs = []
        for img in imgs:
            # packed side may arrive as a CHW numpy array or tensor
            t = torch.as_tensor(np.ascontiguousarray(img)).to(self.device)
            if self._channel_conversion:
                t = t[[2, 1, 0], ...]
            t = t.float()
            if self._enable_normalize:
                t = (t - self.mean) / self.std
            outs.append(t)
        return stack_batch(outs, self.pad_size_divisor, self.pad_value)

    def forward(self, data: dict, training: bool = False):
        if data is None or not any(
                k in data for k in ("img_sonar", "img_camera")):
            # fall back to the single-image path (e.g. plain YOLOX configs)
            return super().forward(data, training=training)

        data = self.cast_data(data)
        inputs: Dict[str, Optional[torch.Tensor]] = {}

        def _opt(key: str) -> Optional[torch.Tensor]:
            imgs = data.get(key)
            if imgs is None or len(imgs) == 0:
                return None
            imgs = list(imgs)
            if any(img is None for img in imgs):
                # mixed batch (some samples dropped this stream, e.g. a
                # per-sample D∞ draw): tensors cannot be stacked with
                # holes, so the stream is dropped for the whole batch.
                # Training keeps batches dense via dinf_mode="zero";
                # test runs use batch_size=1, so a drop stays per-sample.
                return None
            return self._process_stream(imgs)

        inputs["sonar"] = _opt("img_sonar")
        inputs["camera"] = _opt("img_camera")
        inputs["camera_clear"] = _opt("img_camera_clear")

        samples: Dict[str, List[DetDataSample]] = {}
        for key in ("camera", "sonar"):
            ds = data.get(f"data_samples_{key}")
            if ds is None:
                continue
            ds = [d for d in ds if d is not None]  # D∞-dropped sides
            if not ds:
                continue
            # match DetDataPreprocessor: heads/assigners want plain tensors,
            # and metainfo must reflect the padded batch shape
            stream = inputs.get(key)
            if stream is not None:
                shape = tuple(stream.shape[-2:])
                for s in ds:
                    s.set_metainfo({"batch_input_shape": shape,
                                    "pad_shape": shape})
            if self.boxtype2tensor:
                samplelist_boxtype2tensor(ds)
            samples[key] = ds
        return dict(inputs=inputs, data_samples=samples)
