from typing import Literal, Union

# Dataset types
DatasetGroup = Literal["train", "val", "test"]

# Model output types
ClassificationTarget = Literal["species", "breed", "bbox"]
SegmentationTarget = Literal["segmentation"]
IsAnimalTarget = Literal["is_animal"]
PretrainTarget = Union[ClassificationTarget, IsAnimalTarget]
ModelTarget = Union[ClassificationTarget, SegmentationTarget, IsAnimalTarget]


