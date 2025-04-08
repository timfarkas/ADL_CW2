import os

import data
from models import (
    BboxHead,
    CAMManager,
    CNNBackbone,
    ClassifierHead,
    ResNetBackbone,
    TrainedModel,
)
from pre_training import (
    Trainer,
    NUM_SPECIES,
    NUM_BREEDS,
    checkpoints_dir,
)

from utils import save_tensor_dataset


cam_types = ["GradCAM", "ScoreCAM", "AblationCAM"]
cam_folder = "cam_data"

#TODO: Set the right list of checkpoints to generate
checkpoint_dicts = [
    {
        "model_path": "cnn_species",
        "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN")],
        "epoch": 2,
    },
    {
        "model_path": "cnn_breed",
        "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN")],
        "epoch": 2,
    },
    {
        "model_path": "cnn_breed_species",
        "heads": [
            ClassifierHead(NUM_BREEDS, adapter="CNN"),
            ClassifierHead(NUM_SPECIES, adapter="CNN"),
        ],
        "epoch": 2,
    },
    {
        "model_path": "cnn_bbox",
        "heads": [BboxHead(adapter="CNN")],
        "epoch": 2,
    },
    {
        "model_path": "res_species",
        "size": 18,
        "heads": [ClassifierHead(NUM_SPECIES, adapter="Res")],
        "epoch": 5,
    },
    {
        "model_path": "res_species",
        "size": 18,
        "heads": [ClassifierHead(NUM_SPECIES, adapter="Res")],
        "epoch": 15,
    },
]

if __name__ == "__main__":
    dataloader_train, _, _ = data.create_dataloaders(
        batch_size=64,
        resize_size=64,
        target_type=["species", "breed", "bbox", "segmentation"],
        lazy_loading=True,
        shuffle=False,
    )

    print("\n------------------------ Generating CAMS ---------------------\n")

    for checkpoint in checkpoint_dicts:
        if not checkpoint.get("model_path"):
            print("Checkpoint does not have a model path.")
            continue

        path = checkpoint["model_path"]
        path_parts = path.split("_")
        checkpoint_path = os.path.join(checkpoints_dir, path)

        trainer = Trainer()
        if path_parts[0] == "cnn":
            backbone = CNNBackbone()
            trainer.set_model(backbone, checkpoint["heads"], checkpoint_path)
        elif path_parts[0] == "res":
            size = checkpoint["size"]
            backbone = ResNetBackbone(model_type=f"resnet" + size)
            [head.change_adapter("res" + size) for head in checkpoint["heads"]]
            trainer.set_model(
                backbone, checkpoint["heads"], checkpoint_path + "_" + size
            )

        checkpoint_file_path = trainer.checkpoint_path(checkpoint["epoch"])
        trainer.load_checkpoint(checkpoint_file_path)

        for head_index, head in enumerate(trainer.heads):
            model = TrainedModel(backbone=trainer.backbone, head=head)

            for cam in cam_types:
                target_type = path_parts[head_index + 1]

                print(f"Generating {cam} for {path} head {target_type}")

                cam_manager = CAMManager(
                    model=model,
                    dataloader=dataloader_train,
                    target_type=target_type,
                    method=cam,
                )
                save_tensor_dataset(
                    cam_manager.dataset,
                    os.path.join(
                        cam_folder,
                        f"{cam + '_'}{target_type + '_' if len(trainer.heads) > 1 else ''}{os.path.basename(checkpoint_file_path)}",
                    ),
                )
