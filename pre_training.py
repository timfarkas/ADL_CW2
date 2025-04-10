from data import OxfordPetDataset
import data
import json
import os        
import mixed_data  # Added import for mixed_data module to use mixed dataset

from models import ResNetBackbone, CNNBackbone, BboxHead, ClassifierHead
import torch 
import torch.nn as nn
import sys
import io
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import compute_accuracy, computeBBoxIoU, convertVOCBBoxFormatToAnchorFormat

### Num Inputs:
#       Breed:                  37
#       Species:                2
#       Bbox:                   256x256       
#       Breed + Species??:      39
#       Breed + Bbox:           37 + 256x256
#       Species + Bbox:         2 + 256x256
#       Breed+Species+Bbox?:    39 + 256x256

# Backbone:
#   CNN
#   ResNet
# Head:
#   Bbox Head
#   Classifier Head

class Trainer:
    """
    A trainer class for managing model training with multiple heads.
    
    This class handles the setup and training of a backbone network with multiple
    task-specific heads, managing data loaders, loss functions, and optimization.
    """

    def __init__(self, log_dir="logs", log_file="training.json"):
        """Initialize the Trainer with default None values for all attributes."""
        self.backbone = None
        self.heads = None
        self.model_path = None
        self.loss_functions = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        self.eval_functions = None
        self.eval_fn_names = None

        self.log_dir = log_dir
        self.log_file = log_file

    def __repr__(self):
        """Return a string representation of the Trainer instance."""
        return (f"Trainer(\n"
                f"  backbone: {self.backbone.__class__.__name__ if self.backbone else None},\n"
                f"  heads: {[head.__class__.__name__ for head in self.heads] if self.heads else None},\n"
                f"  model_path: {self.model_path},\n"
                f"  loss_functions: {[fn.__class__.__name__ if hasattr(fn, '__class__') else type(fn).__name__ for fn in self.loss_functions] if self.loss_functions else None},\n"
                f"  optimizer: {self.optimizer.__class__.__name__ if self.optimizer else None},\n"
                f"  train_loader: {type(self.train_loader).__name__ if self.train_loader else None},\n"
                f"  val_loader: {type(self.val_loader).__name__ if self.val_loader else None}\n"
                f")")

    def set_model(self, backbone: nn.Module, heads: list[nn.Module], model_path: str):
        """
        Set the model components and save path.
        
        Args:
            backbone: The backbone neural network
            heads: List of task-specific heads that process backbone features
            model_path: Path where the trained model will be saved
        
        Raises:
            AssertionError: If inputs have invalid types
        """
        assert isinstance(backbone, nn.Module), f"Backbone had invalid type ({type(backbone)})"
        assert isinstance(heads, list), f"Heads must be a list, got {type(heads)}"
        for i, head in enumerate(heads):
            assert isinstance(head, nn.Module), f"Head at index {i} had invalid type ({type(head)})"
        assert isinstance(model_path, str), f"model_path must be a string, got {type(model_path)}"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.backbone = backbone
        self.heads = heads
        self.model_path = model_path

    def save_checkpoint(self, epoch, additional_info=None):
        """
        Save a checkpoint of the current model state.
        
        Args:
            epoch: Current training epoch
            additional_info: Optional dictionary with additional information to save
        
        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'heads_state_dict': [head.state_dict() for head in self.heads],
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None
        }
        
        if additional_info is not None:
            checkpoint.update(additional_info)
        
        checkpoint_path = f"{os.path.splitext(self.model_path)[0]}_checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Dictionary containing additional information from the checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            AssertionError: If model components aren't set before loading
        """
        assert self.backbone is not None, "Backbone must be set before loading checkpoint"
        assert self.heads is not None, "Heads must be set before loading checkpoint"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        
        assert len(self.heads) == len(checkpoint['heads_state_dict']), "Number of heads doesn't match checkpoint"
        for head, state_dict in zip(self.heads, checkpoint['heads_state_dict']):
            head.load_state_dict(state_dict)
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        additional_info = {k: v for k, v in checkpoint.items() 
                          if k not in ['backbone_state_dict', 'heads_state_dict', 'optimizer_state_dict']}
        
        print(f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
        return additional_info

    def set_loss_functions(self, loss_functions: list[callable]):
        """
        Set the loss functions for each head.
        
        Args:
            loss_functions: List of loss functions corresponding to each head
        """
        self.loss_functions = loss_functions 

    def set_optimizer(self, learning_rate: float, weight_decay: float):
        """
        Set up the optimizer with all trainable parameters.
        
        Args:
            learning_rate: Learning rate for the optimizer
            
        Raises:
            AssertionError: If heads are not set
        """
        assert self.heads is not None, "Heads must be set before setting optimizer"
        all_params = list(self.backbone.parameters())
        for head in self.heads:
            all_params.extend(head.parameters())

        self.optimizer = optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)

    def set_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Set the data loaders for training and validation.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Raises:
            AssertionError: If inputs have invalid types
        """
        assert isinstance(train_loader, DataLoader), f"train_loader must be a DataLoader, got {type(train_loader)}"
        assert isinstance(val_loader, DataLoader), f"val_loader must be a DataLoader, got {type(val_loader)}"
        
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_eval_functions(self, eval_functions: list[callable], fn_names: list[str]):
        """
        Set the eval function for each head.
        Each eval function has to be of the form eval_fn(predictions, targets) -> float.
        
        Args:
            eval_functions: List of evaluation functions for each head
            fn_names: List of names for each evaluation function
        
        Raises:
            AssertionError: If lists have different lengths
        """
        assert len(eval_functions) == len(fn_names), "Number of evaluation functions must match number of function names"
        self.eval_functions = eval_functions
        self.eval_fn_names = fn_names

    def _forward_pass(self, x: torch.Tensor) -> list:
        """
        Perform a forward pass through the backbone and all heads.
        
        Args:
            x: Input tensor
            
        Returns:
            List of outputs from each head
        """
        features = self.backbone(x)
        outputs = [head(features) for head in self.heads]
        return outputs

    def log_performance(self, model_name, epoch, metrics_list):
        """
        Log performance metrics to a JSON file.
        
        Args:
            model_name: Name of the model being trained
            epoch: Current epoch number
            metrics_list: List of tuples (metric_name, metric_value) to log
        """
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, self.log_file)
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {}
        
        if model_name not in log_data:
            log_data[model_name] = {}

        epoch_key = str(epoch)
        if epoch_key not in log_data[model_name]:
            log_data[model_name][epoch_key] = {}
        
        for metric_name, metric_value in metrics_list:
            log_data[model_name][epoch_key][metric_name] = metric_value
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)

    def fit_sgd(self, num_epochs: int = 20, learning_rate: float = 3e-4, 
                checkpoint_interval: int = 5, device: str = None) -> None:
        """
        Trains model heads.
        
        Args:
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for the optimizer (if not already set)
            checkpoint_interval: Number of epochs between checkpoints (0 to disable)
            device: Device to run the training on ('cuda', 'cpu', etc.)
        """
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Training {self.model_path} for {num_epochs} on {device}....")
        if learning_rate is not None and self.optimizer is None:
            self.set_optimizer(learning_rate)
        
        assert self.optimizer is not None, "Optimizer must be set before training"
        assert self.train_loader is not None, "Train loader must be set before training"
        assert self.val_loader is not None, "Validation loader must be set before training"
        assert self.loss_functions is not None, "Loss functions must be set before training"
        
        print(self.__repr__())
        print("--- Training Start ---")
        self.backbone = self.backbone.to(device)
        self.heads = [head.to(device) for head in self.heads]
        
        batch_count = 0
        
        for epoch in range(num_epochs):
            self.backbone.train()
            for head in self.heads:
                head.train()
                
            train_sample_count = 0
            train_epoch_loss_sum = 0
            train_epoch_head_evals_sum = [0 for _ in self.heads]
            train_epoch_head_losses_sum = [0 for _ in self.heads]
            # Updated: Use train_valid_sample_counts instead of train_bbox_sample_counts to handle both head types
            train_valid_sample_counts = [0 for _ in self.heads]  # Track valid samples for each head

            print(f"\nTrain {epoch+1}/{num_epochs} ", end="")
            for i, (images, labels) in enumerate(self.train_loader):
                labels = [labels] if isinstance(labels, torch.Tensor) else labels
                labels = [labels[key] for key in labels.keys()] if isinstance(labels, dict) else labels

                if i % max(1, len(self.train_loader) // 10) == 0:
                    print(f"|", end="", flush=True)
                
                images = images.to(device)
                labels = [label.to(device) for label in labels]

                batch_count += 1
                self.optimizer.zero_grad()
                
                head_outputs = self._forward_pass(images) 
                
                # Updated: Compute losses with masking for both BboxHead and ClassifierHead
                losses = []
                for head, loss_fn, head_output, label in zip(self.heads, self.loss_functions, head_outputs, labels):
                    if isinstance(head, BboxHead):
                        per_sample_loss = loss_fn(head_output, label).mean(dim=1)
                        mask = (label.sum(dim=1) != 0).float()
                        if mask.sum() > 0:
                            loss = (per_sample_loss * mask).sum() / mask.sum()
                        else:
                            loss = torch.tensor(0.0, device=device)
                    else:  # ClassifierHead uses ignore_index=-1 in loss_fn
                        loss = loss_fn(head_output, label)
                    losses.append(loss)
                total_batch_loss = sum(losses)
                total_batch_loss.backward()
                
                self.optimizer.step()
                
                batch_size = len(images)
                train_sample_count += batch_size
                
                running_loss = total_batch_loss.item() * batch_size
                train_epoch_loss_sum += running_loss
                
                # Updated: Compute head losses with valid sample counts
                head_losses = []
                for j, (head, loss, label) in enumerate(zip(self.heads, losses, labels)):
                    if isinstance(head, ClassifierHead):
                        mask = label != -1
                        num_valid = mask.sum().item()
                        head_losses.append(loss.item() * num_valid)
                        train_valid_sample_counts[j] += num_valid
                    elif isinstance(head, BboxHead):
                        mask = (label.sum(dim=1) != 0)
                        num_valid = mask.sum().item()
                        head_losses.append(loss.item() * num_valid)
                        train_valid_sample_counts[j] += num_valid
                    else:
                        head_losses.append(loss.item() * batch_size)
                        train_valid_sample_counts[j] += batch_size
                train_epoch_head_losses_sum = [epoch_sum + batch_loss for epoch_sum, batch_loss in zip(train_epoch_head_losses_sum, head_losses)]
                
                # Updated: Compute evaluation metrics with masking
                if self.eval_functions is not None:
                    batch_head_evals = []
                    for j, (eval_fn, head, head_output, label) in enumerate(zip(self.eval_functions, self.heads, head_outputs, labels)):
                        if isinstance(head, ClassifierHead):
                            mask = label != -1
                        elif isinstance(head, BboxHead):
                            mask = (label.sum(dim=1) != 0)
                        else:
                            mask = torch.ones(label.size(0), dtype=torch.bool, device=device)
                        if mask.any():
                            head_output_masked = head_output[mask]
                            label_masked = label[mask]
                            eval_value = eval_fn(head_output_masked, label_masked)
                            batch_eval = float(eval_value) * mask.sum().item()
                        else:
                            batch_eval = 0.0
                        batch_head_evals.append(batch_eval)
                    train_epoch_head_evals_sum = [epoch_sum + batch_eval for epoch_sum, batch_eval in zip(train_epoch_head_evals_sum, batch_head_evals)]

            print(f"\nEpoch:{epoch + 1}/{num_epochs}, Train Loss:{train_epoch_loss_sum / train_sample_count:.4f}")
            
            metrics_to_log = [("train_loss", train_epoch_loss_sum / train_sample_count)]
            
            # Updated: Use train_valid_sample_counts for averaging
            for i, head in enumerate(self.heads):
                head_loss = train_epoch_head_losses_sum[i] / train_valid_sample_counts[i] if train_valid_sample_counts[i] > 0 else 0.0
                log_str = f"   Train {head.name} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"train_{head.name}_loss", head_loss))
                
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    metric_value = train_epoch_head_evals_sum[i] / train_valid_sample_counts[i] if train_valid_sample_counts[i] > 0 else 0.0
                    log_str += f", {metric_name}: {metric_value:.4f}"
                    metrics_to_log.append((f"train_{head.name}_{metric_name}", metric_value))
                
                print(log_str)
            
            self.backbone.eval()
            for head in self.heads:
                head.eval()
                
            val_sample_count = 0
            val_epoch_loss_sum = 0
            val_epoch_head_evals_sum = [0 for _ in self.heads]
            val_epoch_head_losses_sum = [0 for _ in self.heads]
            # Updated: Use val_valid_sample_counts
            val_valid_sample_counts = [0 for _ in self.heads]

            with torch.no_grad():
                for images, labels in self.val_loader:
                    labels = [labels] if isinstance(labels, torch.Tensor) else labels
                    labels = [labels[key] for key in labels.keys()] if isinstance(labels, dict) else labels

                    images = images.to(device)
                    labels = [label.to(device) for label in labels]
                    
                    head_outputs = self._forward_pass(images)
                    
                    losses = []
                    for head, loss_fn, head_output, label in zip(self.heads, self.loss_functions, head_outputs, labels):
                        if isinstance(head, BboxHead):
                            per_sample_loss = loss_fn(head_output, label).mean(dim=1)
                            mask = (label.sum(dim=1) != 0).float()
                            if mask.sum() > 0:
                                loss = (per_sample_loss * mask).sum() / mask.sum()
                            else:
                                loss = torch.tensor(0.0, device=device)
                        else:
                            loss = loss_fn(head_output, label)
                        losses.append(loss)
                    total_batch_loss = sum(losses)
                    
                    batch_size = len(images)
                    val_sample_count += batch_size
                    
                    val_running_loss = total_batch_loss.item() * batch_size
                    val_epoch_loss_sum += val_running_loss
                    
                    # Updated: Compute head losses with valid sample counts
                    head_losses = []
                    for j, (head, loss, label) in enumerate(zip(self.heads, losses, labels)):
                        if isinstance(head, ClassifierHead):
                            mask = label != -1
                            num_valid = mask.sum().item()
                            head_losses.append(loss.item() * num_valid)
                            val_valid_sample_counts[j] += num_valid
                        elif isinstance(head, BboxHead):
                            mask = (label.sum(dim=1) != 0)
                            num_valid = mask.sum().item()
                            head_losses.append(loss.item() * num_valid)
                            val_valid_sample_counts[j] += num_valid
                        else:
                            head_losses.append(loss.item() * batch_size)
                            val_valid_sample_counts[j] += batch_size
                    val_epoch_head_losses_sum = [epoch_sum + batch_loss for epoch_sum, batch_loss in zip(val_epoch_head_losses_sum, head_losses)]
                    
                    # Updated: Compute evaluation metrics with masking
                    if self.eval_functions is not None:
                        val_batch_head_evals = []
                        for j, (eval_fn, head, head_output, label) in enumerate(zip(self.eval_functions, self.heads, head_outputs, labels)):
                            if isinstance(head, ClassifierHead):
                                mask = label != -1
                            elif isinstance(head, BboxHead):
                                mask = (label.sum(dim=1) != 0)
                            else:
                                mask = torch.ones(label.size(0), dtype=torch.bool, device=device)
                            if mask.any():
                                head_output_masked = head_output[mask]
                                label_masked = label[mask]
                                eval_value = eval_fn(head_output_masked, label_masked)
                                batch_eval = float(eval_value) * mask.sum().item()
                            else:
                                batch_eval = 0.0
                            val_batch_head_evals.append(batch_eval)
                        val_epoch_head_evals_sum = [epoch_sum + batch_eval for epoch_sum, batch_eval in zip(val_epoch_head_evals_sum, val_batch_head_evals)]
            
            print(f"Epoch:{epoch + 1}/{num_epochs}, Val Loss:{val_epoch_loss_sum / val_sample_count:.4f}")
            metrics_to_log.append(("val_loss", val_epoch_loss_sum / val_sample_count))
            
            # Updated: Use val_valid_sample_counts for averaging
            for i, head in enumerate(self.heads):
                head_loss = val_epoch_head_losses_sum[i] / val_valid_sample_counts[i] if val_valid_sample_counts[i] > 0 else 0.0
                log_str = f"  Val {head.name} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"val_{head.name}_loss", head_loss))
                
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    metric_value = val_epoch_head_evals_sum[i] / val_valid_sample_counts[i] if val_valid_sample_counts[i] > 0 else 0.0
                    log_str += f", {metric_name}: {metric_value:.4f}"
                    metrics_to_log.append((f"val_{head.name}_{metric_name}", metric_value))
                
                print(log_str)
            
            model_name = os.path.basename(self.model_path).split('.')[0]
            self.log_performance(model_name, epoch + 1, metrics_to_log)

            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch=epoch + 1, additional_info={
                    'train_loss': train_epoch_loss_sum / train_sample_count,
                    'val_loss': val_epoch_loss_sum / val_sample_count
                })
        
        print("Training finished!")
        self.save_checkpoint(epoch=num_epochs, additional_info={
            'train_loss': train_epoch_loss_sum / train_sample_count,
            'val_loss': val_epoch_loss_sum / val_sample_count,
            'is_final_model': True
        })

        print(f"Final model saved to {self.model_path}")
        print(f"Number of parameters: {sum(p.numel() for p in self.backbone.parameters()) + sum(sum(p.numel() for p in head.parameters()) for head in self.heads)}")

def convert_and_get_IoU(outputs, targets) -> float:
    outputs = convertVOCBBoxFormatToAnchorFormat(outputs)
    targets = convertVOCBBoxFormatToAnchorFormat(targets)
    iou = computeBBoxIoU(outputs, targets)
    return iou

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")

    print("Testing models, loader and devices.")
    backbone = CNNBackbone().to(device)
    res_backbone = ResNetBackbone().to(device)
    cnn_head = BboxHead(adapter="cnn").to(device)
    res_head = BboxHead(adapter="res").to(device)
    res_class_head = ClassifierHead(adapter="res", num_classes=37).to(device)
    res_class_head2 = ClassifierHead(adapter="res", num_classes=37).to(device)
    cnn_class_head = ClassifierHead(adapter="cnn", num_classes=37).to(device)

    cnn_params = sum(p.numel() for p in backbone.parameters())
    print(f"CNN Backbone parameters: {cnn_params:,}")
    
    res_params = sum(p.numel() for p in res_backbone.parameters())
    print(f"ResNet Backbone parameters: {res_params:,}")

    loader, _, _ = data.create_dataloaders()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        features = backbone(images)
        features2 = res_backbone(images)
        
        predictions = cnn_head(features)
        predictions2 = res_head(features2)

        predictions3 = res_class_head(features2)
        predictions4 = cnn_class_head(features)
        print("All tests passed.")
        break

    print("\n\nPreparing pre-training sweep...")

    # Updated: Use ignore_index=-1 for classification to handle background images
    cel_fn = nn.CrossEntropyLoss(ignore_index=-1)
    mse_fn = nn.MSELoss(reduction='none')

    checkpoints_dir = "checkpoints"
    NUM_SPECIES = 2
    NUM_BREEDS = 37

    run_dicts = [
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_species"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy],
            "eval_function_names": ["Acc"],
            "loss_functions": [cel_fn],
            "loader_targets": ["species"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_breed"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy],
            "eval_function_names": ["Acc"],
            "loss_functions": [cel_fn],
            "loader_targets": ["breed"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_bbox"),
            "heads": [BboxHead(adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [convert_and_get_IoU],
            "eval_function_names": ["IoU"], 
            "loss_functions": [mse_fn],
            "loader_targets": ["bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_breed_species"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN"), ClassifierHead(NUM_SPECIES, adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy, compute_accuracy],
            "eval_function_names": ["Acc", "Acc"],
            "loss_functions": [cel_fn, cel_fn],
            "loader_targets": ["breed", "species"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_breed_bbox"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="CNN"), BboxHead(adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "IoU"], 
            "loss_functions": [cel_fn, mse_fn], 
            "loader_targets": ["breed", "bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_species_bbox"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN"), BboxHead(adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "IoU"],
            "loss_functions": [cel_fn, mse_fn], 
            "loader_targets": ["species", "bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_species_breed_bbox"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN"), ClassifierHead(NUM_BREEDS, adapter="CNN"), BboxHead(adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy, compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "Acc", "IoU"], 
            "loss_functions": [cel_fn, cel_fn, mse_fn],
            "loader_targets": ["species", "breed", "bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_species"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy],
            "eval_function_names": ["Acc"], 
            "loss_functions": [cel_fn],
            "loader_targets": ["species"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_breed"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy],
            "eval_function_names": ["Acc"],
            "loss_functions": [cel_fn],
            "loader_targets": ["breed"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_bbox"),
            "heads": [BboxHead(adapter="Res")],
            "backbone": "res",
            "eval_functions": [convert_and_get_IoU],
            "eval_function_names": ["IoU"],
            "loss_functions": [mse_fn], 
            "loader_targets": ["bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_breed_species"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="Res"), ClassifierHead(NUM_SPECIES, adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy, compute_accuracy],
            "eval_function_names": ["Acc", "Acc"],
            "loss_functions": [cel_fn, cel_fn],
            "loader_targets": ["breed", "species"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_breed_bbox"),
            "heads": [ClassifierHead(NUM_BREEDS, adapter="Res"), BboxHead(adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "IoU"],
            "loss_functions": [cel_fn, mse_fn],
            "loader_targets": ["breed", "bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_species_bbox"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="Res"), BboxHead(adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "IoU"],
            "loss_functions": [cel_fn, mse_fn],
            "loader_targets": ["species", "bbox"]
        },
        {
            "model_path": os.path.join(checkpoints_dir, "res_species_breed_bbox"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="Res"), ClassifierHead(NUM_BREEDS, adapter="Res"), BboxHead(adapter="Res")],
            "backbone": "res",
            "eval_functions": [compute_accuracy, compute_accuracy, convert_and_get_IoU],
            "eval_function_names": ["Acc", "Acc", "IoU"],
            "loss_functions": [cel_fn, cel_fn, mse_fn],
            "loader_targets": ["species", "breed", "bbox"]
        }
    ]

    print(f"Starting {len(run_dicts)} runs....\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset = OxfordPetDataset() 
    
    batch_size = 64 
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 15 
    
    print(f"Using {device}")

    for i, run_dict in enumerate(run_dicts):
        model_path = run_dict['model_path']        
        print(f"Starting training run {i+1}, {os.path.basename(model_path)}...")

        print("Setting up trainer...")

        # Updated: Use "pretraining_mixed.json" to distinguish logs
        trainer = Trainer(log_dir="logs", log_file="pretraining_mixed.json")

        trainer.set_eval_functions(run_dict["eval_functions"], run_dict["eval_function_names"])
       
        # Updated: Use mixed_data.create_mixed_dataloaders for mixed dataset with augmentation
        train_loader, val_loader, _ = mixed_data.create_mixed_dataloaders(
            batch_size=batch_size,
            target_type=run_dict["loader_targets"],
            lazy_loading=False,
            use_augmentation=True,
            data_directory="oxford_pet_data",
            bg_directory="bg-20k/train",  # Kaggle landscape images
            mixing_ratio=5,
            bg_label=-1,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
            normalize_bbox=True
        )
        trainer.set_loaders(train_loader, val_loader) 
        trainer.set_loss_functions(run_dict['loss_functions'])

        backbone = None

        if run_dict["backbone"] == "cnn":
            backbone = CNNBackbone()
            # Updated: Append "_mixed" to model_path
            trainer.set_model(backbone, run_dict['heads'], run_dict['model_path'] + "_mixed")
            trainer.set_optimizer(learning_rate, weight_decay)
            print("Trainer set up successfully!")
            trainer.fit_sgd(device=device)
        elif run_dict['backbone'] == "res":
            for size in ['18', '50']:
                backbone = ResNetBackbone(model_type=f"resnet{size}")
                [head.change_adapter(f"res{size}") for head in run_dict['heads']]
                # Updated: Append "_mixed_" + size to model_path
                trainer.set_model(backbone, run_dict['heads'], run_dict['model_path'] + "_mixed_" + size)
                trainer.set_optimizer(learning_rate, weight_decay)
                print("Trainer set up successfully!")
                trainer.fit_sgd(device=device)        