## Version 0.0 Dataset API, includes FAST-VQA and its variants
from .basic_datasets import (
    FastVQAPlusPlusDataset,
    FragmentVideoDataset,
    FragmentImageDataset,
    ResizedVideoDataset,
    ResizedImageDataset,
    CroppedVideoDataset,
    CroppedImageDataset,
    SampleFrames,
    FragmentSampleFrames,
)

## Version 1.0 Dataset API, includes DiViDe VQA and its variants
from .fusion_datasets import get_spatial_fragments_and_motion_resize, FusionDataset


__all__ = [
    "FragmentVideoDataset",
    "FragmentImageDataset",
    "ResizedVideoDataset",
    "ResizedImageDataset",
    "CroppedVideoDataset",
    "CroppedImageDataset",
    "SampleFrames",
    "FragmentSampleFrames",
    "FusionDataset",
]