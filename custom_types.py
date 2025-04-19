from typing import Literal, Union

# Dataset types
DatasetGroup = Literal["train", "val", "test"]

# Model output types
ClassificationTarget = Literal["species", "breed", "bbox"]
SegmentationTarget = Literal["segmentation"]
IsAnimalTarget = Literal["is_animal"]
PretrainTarget = Union[ClassificationTarget, IsAnimalTarget]
ModelTarget = Union[ClassificationTarget, SegmentationTarget, IsAnimalTarget]

# Target numbers
ClassificationTargetNumbers = {
    "species": 2,
    "breed": 37,
    "is_animal": 2,
    "bbox": 4,
}

# Model adapter types
CnnAdapter = Literal["cnn"]
ResNetAdapter = Literal["res18", "res50", "res101"]
AdapterType = Union[CnnAdapter, ResNetAdapter]

# Adapter numbers
AdapterNumbers = {
    "cnn": 256,
    "res18": 512,
    "res50": 2048,
    "res101": 2048,
}
