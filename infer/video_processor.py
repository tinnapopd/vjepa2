from transformers.image_utils import (  # type: ignore
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    PILImageResampling,
)
from transformers.processing_utils import Unpack, VideosKwargs  # type: ignore
from transformers.video_processing_utils import BaseVideoProcessor  # type: ignore


class VJEPA21VideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": int(384 * 256 / 224)}
    crop_size = 384
    do_resize = True
    do_rescale = True
    do_center_crop = True
    do_normalize = True

    def __init__(self, **kwargs: Unpack[VideosKwargs]):
        crop_size = kwargs.get("crop_size", 384)
        if not isinstance(crop_size, int):
            if not isinstance(crop_size, dict) or "height" not in crop_size:
                raise ValueError(
                    "crop_size must be an integer or a dictionary with a 'height' key"
                )
            crop_size = crop_size["height"]
        resize_size = int(crop_size * 256 / 224)
        kwargs["size"] = {"shortest_edge": resize_size}
        super().__init__(**kwargs)


__all__ = ["VJEPA21VideoProcessor"]
