import os
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Import project modules
from models import CNNBackbone, ResNetBackbone, ClassifierHead, BboxHead
from pre_training import Trainer
from custom_loader import create_dataloaders
from custom_data import OxfordPetDataset
from utils import compute_accuracy  # assuming this is used for classification

# Random seed for reproducibility
random.seed(27)
torch.manual_seed(27)

# Candidate backbones (subject to change)
backbone_candidates = {
    "CNN": CNNBackbone,
    "ResNet": lambda: ResNetBackbone(pretrained=True)  # pretrained weights for ResNet
}

# Define candidate heads for classification tasks (can extend to include BboxHead for bbox tasks)
# In this example, we include two tasks: "species" and "breed"
# Adjust num_classes based on the task requirements.
classification_tasks = [
    {"task": "species", "num_classes": 2},
    {"task": "breed", "num_classes": 37}
]

# Candidate hyperparameters (subject to change)
learning_rates = [1e-3, 3e-4, 1e-4]
num_epochs = 5      # For demonstration; increase for full training runs.
batch_size = 16

# Create the dataset and corresponding dataloaders.
dataset = OxfordPetDataset()
train_loader, val_loader, _ = create_dataloaders(
    dataset,
    batch_size=batch_size,
    target_type=["class"]  # using image-level labels for classification
)

# List of configuration dictionaries.
configurations = []
for backbone_name, BackboneClass in backbone_candidates.items():
    for lr in learning_rates:
        for task in classification_tasks:
            # Choose the adapter string based on backbone type
            adapter = backbone_name  # This assumes head adapters match backbone names (e.g., "CNN" or "ResNet")
            
            # Instantiate the head (here using ClassifierHead)
            head = ClassifierHead(num_classes=task["num_classes"], adapter=adapter)
            # Instantiate the backbone
            backbone = BackboneClass()
            # Build a unique model_path for checkpointing.
            model_path = os.path.join("checkpoints", f"{backbone_name}_{task['task']}_lr{lr}")
            
            # Create configuration dictionary.
            config = {
                "backbone": backbone,
                "heads": [head],
                "model_path": model_path,
                "learning_rate": lr,
                "target_type": [task["task"]],  # e.g., ["species"] or ["breed"]
                "eval_functions": [compute_accuracy],  # using our compute_accuracy for classification
                "eval_function_names": ["Acc"],
                "loss_functions": [nn.CrossEntropyLoss()]
            }
            configurations.append(config)

# Optionally, sample a subset of configurations if the total number is large.
sampled_configs = random.sample(configurations, min(len(configurations), 5))

# Determine the device for training on Apple Silicon.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Run training for each sampled configuration.
for config in sampled_configs:
    print(f"\n=== Running configuration: {config['model_path']} (LR={config['learning_rate']}) ===")
    
    # Instantiate Trainer and configure it.
    trainer = Trainer()
    trainer.set_model(config["backbone"], config["heads"], config["model_path"])
    trainer.set_optimizer(config["learning_rate"])
    trainer.set_loss_functions(config["loss_functions"])
    trainer.set_eval_functions(config["eval_functions"], config["eval_function_names"])
    trainer.set_loaders(train_loader, val_loader)
    
    # Start training (using a reduced number of epochs for a demo)
    trainer.fit_sgd(num_epochs=num_epochs, device=device)
    
    print(f"Finished training configuration: {config['model_path']}")
