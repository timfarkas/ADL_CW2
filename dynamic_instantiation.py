import os
import random
import torch
import torch.nn as nn
from models import CNNBackbone, ResNetBackbone, ClassifierHead
from pre_training import Trainer
from data import create_dataloaders
from data import OxfordPetDataset
from utils import compute_accuracy

# Set random seeds for reproducibility
random.seed(27)
torch.manual_seed(27)

# Define the current backbones: CNN and ResNet
backbone_candidates = {
    "CNN": CNNBackbone,
    "ResNet": lambda: ResNetBackbone(pretrained=True),
}

# Define classification tasks
classification_tasks = [
    {"task": "species", "num_classes": 2},  # Cat or dog
    {"task": "breed", "num_classes": 37},  # 37 breed classes
]

# Hyperparameters
learning_rates = [1e-3, 3e-4, 1e-4]
num_epochs = 5  # For demonstration; increase for real training
batch_size = 16

# Initialize dataset and create task-specific dataloaders
dataset = OxfordPetDataset()
train_loader_breed, val_loader_breed, _ = create_dataloaders(
    dataset, batch_size=batch_size, target_type=["breed"]
)
train_loader_species, val_loader_species, _ = create_dataloaders(
    dataset, batch_size=batch_size, target_type=["species"]
)

# Generate all possible configurations
configurations = []
for backbone_name, BackboneClass in backbone_candidates.items():
    for lr in learning_rates:
        for task in classification_tasks:
            # Instantiate head with backbone-specific adapter
            head = ClassifierHead(
                num_classes=task["num_classes"], adapter=backbone_name
            )
            backbone = BackboneClass()
            model_path = os.path.join(
                "checkpoints", f"{backbone_name}_{task['task']}_lr{lr}"
            )
            config = {
                "backbone": backbone,
                "heads": [head],
                "model_path": model_path,
                "learning_rate": lr,
                "target_type": [task["task"]],
                "eval_functions": [compute_accuracy],
                "eval_function_names": ["Acc"],
                "loss_functions": [nn.CrossEntropyLoss()],
            }
            configurations.append(config)

# Sample a subset of configurations (up to 5)
sampled_configs = random.sample(configurations, min(len(configurations), 5))

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Training loop
for config in sampled_configs:
    print(
        f"\n=== Running configuration: {config['model_path']} (LR={config['learning_rate']}) ==="
    )
    trainer = Trainer()
    trainer.set_model(config["backbone"], config["heads"], config["model_path"])
    trainer.set_optimizer(config["learning_rate"])  # Uses AdamW by default
    trainer.set_loss_functions(config["loss_functions"])
    trainer.set_eval_functions(config["eval_functions"], config["eval_function_names"])
    # Select dataloader based on task
    if config["target_type"][0] == "breed":
        trainer.set_loaders(train_loader_breed, val_loader_breed)
    elif config["target_type"][0] == "species":
        trainer.set_loaders(train_loader_species, val_loader_species)
    trainer.fit_sgd(
        num_epochs=num_epochs, device=device
    )  # Note: Uses AdamW despite name
    print(f"Finished training configuration: {config['model_path']}")
