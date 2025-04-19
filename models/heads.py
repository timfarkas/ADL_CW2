from torch import nn

from custom_types import (
    AdapterNumbers,
    AdapterType,
    ClassificationTargetNumbers,
    PretrainTarget,
)


class PretrainHead(nn.Module):
    def __init__(self, adapter: AdapterType, output_type: PretrainTarget):
        super().__init__()
        self.change_adapter(adapter, output_type)

    def change_adapter(self, adapter: AdapterType, output_type: PretrainTarget):
        num_inputs = AdapterNumbers.get(adapter.lower())
        num_outputs = ClassificationTargetNumbers.get(output_type.lower())
        is_bbox = output_type.lower() == "bbox"

        if not num_inputs:
            raise ValueError(f"Unknown adapter type: {adapter}")
        if not num_outputs:
            raise ValueError(f"Unknown output type: {output_type}")

        self.head = nn.Sequential(
            *[
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(num_inputs, num_outputs),
                *([nn.Sigmoid()] if is_bbox else []),  # ← conditional layer
            ]
        )

        self.name = "BBoxHead" if is_bbox else f"ClassifierHead({num_outputs})"
        self.is_bbox = is_bbox

    def forward(self, z):
        return self.head(z)
