import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import data
import mixed_data
from data import OxfordPetDataset
from models import BboxHead, ClassifierHead, CNNBackbone, ResNetBackbone
from utils import compute_accuracy, computeBBoxIoU, convertVOCBBoxFormatToAnchorFormat


class Trainer:
    """
    A trainer class for managing model training with multiple heads, modified for quick logging validation.
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
        self.target_names = None  # Store target-specific names for logging
        self.log_dir = log_dir
        self.log_file = log_file

    def __repr__(self):
        """Return a string representation of the Trainer instance."""
        return (
            f"Trainer(\n"
            f"  backbone: {self.backbone.__class__.__name__ if self.backbone else None},\n"
            f"  heads: {[head.__class__.__name__ for head in self.heads] if self.heads else None},\n"
            f"  model_path: {self.model_path},\n"
            f"  loss_functions: {[fn.__class__.__name__ if hasattr(fn, '__class__') else type(fn).__name__ for fn in self.loss_functions] if self.loss_functions else None},\n"
            f"  optimizer: {self.optimizer.__class__.__name__ if self.optimizer else None},\n"
            f"  train_loader: {type(self.train_loader).__name__ if self.train_loader else None},\n"
            f"  val_loader: {type(self.val_loader).__name__ if self.val_loader else None}\n"
            f")"
        )

    def set_model(self, backbone: nn.Module, heads: list[nn.Module], model_path: str):
        """Set the model components and save path."""
        assert isinstance(backbone, nn.Module), (
            f"Backbone had invalid type ({type(backbone)})"
        )
        assert isinstance(heads, list), f"Heads must be a list, got {type(heads)}"
        for i, head in enumerate(heads):
            assert isinstance(head, nn.Module), (
                f"Head at index {i} had invalid type ({type(head)})"
            )
        assert isinstance(model_path, str), (
            f"model_path must be a string, got {type(model_path)}"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.backbone = backbone
        self.heads = heads
        self.model_path = model_path

    def save_checkpoint(self, epoch, additional_info=None):
        """Save a checkpoint of the current model state."""
        checkpoint = {
            "epoch": epoch,
            "backbone_state_dict": self.backbone.state_dict(),
            "heads_state_dict": [head.state_dict() for head in self.heads],
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
        }
        if additional_info is not None:
            checkpoint.update(additional_info)
        checkpoint_path = (
            f"{os.path.splitext(self.model_path)[0]}_checkpoint_epoch{epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load a model checkpoint."""
        assert self.backbone is not None, (
            "Backbone must be set before loading checkpoint"
        )
        assert self.heads is not None, "Heads must be set before loading checkpoint"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        assert len(self.heads) == len(checkpoint["heads_state_dict"]), (
            "Number of heads doesn't match checkpoint"
        )
        for head, state_dict in zip(self.heads, checkpoint["heads_state_dict"]):
            head.load_state_dict(state_dict)
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        additional_info = {
            k: v
            for k, v in checkpoint.items()
            if k
            not in ["backbone_state_dict", "heads_state_dict", "optimizer_state_dict"]
        }
        print(
            f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})"
        )
        return additional_info

    def set_loss_functions(self, loss_functions: list[callable]):
        """Set the loss functions for each head."""
        self.loss_functions = loss_functions

    def set_optimizer(self, learning_rate: float, weight_decay: float):
        """Set up the optimizer with all trainable parameters."""
        assert self.heads is not None, "Heads must be set before setting optimizer"
        all_params = list(self.backbone.parameters())
        for head in self.heads:
            all_params.extend(head.parameters())
        self.optimizer = optim.AdamW(
            all_params, lr=learning_rate, weight_decay=weight_decay
        )

    def set_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """Set the data loaders for training and validation."""
        assert isinstance(train_loader, DataLoader), (
            f"train_loader must be a DataLoader, got {type(train_loader)}"
        )
        assert isinstance(val_loader, DataLoader), (
            f"val_loader must be a DataLoader, got {type(val_loader)}"
        )
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_eval_functions(self, eval_functions: list[callable], fn_names: list[str]):
        """Set the evaluation functions for each head."""
        assert len(eval_functions) == len(fn_names), (
            "Number of evaluation functions must match number of function names"
        )
        self.eval_functions = eval_functions
        self.eval_fn_names = fn_names

    def set_target_names(self, target_names):
        """Set target names for each head to ensure unique logging."""
        self.target_names = target_names

    def _forward_pass(self, x: torch.Tensor) -> list:
        """Perform a forward pass through the backbone and all heads."""
        features = self.backbone(x)
        outputs = [head(features) for head in self.heads]
        return outputs

    def log_performance(self, model_name, epoch, metrics_list):
        """Log performance metrics to a JSON file."""
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, self.log_file)
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
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
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)

    def fit_sgd(
        self,
        num_epochs: int = 20,
        learning_rate: float = 3e-4,
        checkpoint_interval: int = 5,
        device: str = None,
    ) -> None:
        """Train the model with SGD, modified to limit batches for quick validation."""
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Training {self.model_path} for {num_epochs} on {device}....")
        if learning_rate is not None and self.optimizer is None:
            self.set_optimizer(learning_rate, 1e-4)
        assert self.optimizer is not None, "Optimizer must be set before training"
        assert self.train_loader is not None, "Train loader must be set before training"
        assert self.val_loader is not None, (
            "Validation loader must be set before training"
        )
        assert self.loss_functions is not None, (
            "Loss functions must be set before training"
        )
        assert self.target_names is not None, "Target names must be set before training"
        assert len(self.target_names) == len(self.heads), (
            "Target names must match number of heads"
        )

        print(self.__repr__())
        print("--- Training Start ---")
        self.backbone = self.backbone.to(device)
        self.heads = [head.to(device) for head in self.heads]
        batch_count = 0

        for epoch in range(num_epochs):
            # Training phase
            self.backbone.train()
            for head in self.heads:
                head.train()
            train_sample_count = 0
            train_epoch_loss_sum = 0
            train_epoch_head_evals_sum = [0 for _ in self.heads]
            train_epoch_head_losses_sum = [0 for _ in self.heads]
            train_bbox_sample_counts = [0 for _ in self.heads]

            print(f"\nTrain {epoch + 1}/{num_epochs} ", end="")
            for i, (images, labels) in enumerate(self.train_loader):
                if i >= 2:  # Limit to 2 batches for quick testing
                    break
                labels = [labels] if isinstance(labels, torch.Tensor) else labels
                labels = (
                    [labels[key] for key in labels.keys()]
                    if isinstance(labels, dict)
                    else labels
                )
                if i % max(1, len(self.train_loader) // 10) == 0:
                    print("|", end="", flush=True)
                images = images.to(device)
                labels = [label.to(device) for label in labels]
                batch_count += 1
                self.optimizer.zero_grad()
                head_outputs = self._forward_pass(images)
                losses = []
                for head, loss_fn, head_output, label in zip(
                    self.heads, self.loss_functions, head_outputs, labels
                ):
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
                total_batch_loss.backward()
                self.optimizer.step()

                batch_size = len(images)
                train_sample_count += batch_size
                running_loss = total_batch_loss.item() * batch_size
                train_epoch_loss_sum += running_loss
                head_losses = []
                for j, (head, loss) in enumerate(zip(self.heads, losses)):
                    if isinstance(head, BboxHead):
                        mask = (labels[j].sum(dim=1) != 0).float()
                        head_losses.append(loss.item() * mask.sum().item())
                        train_bbox_sample_counts[j] += mask.sum().item()
                    else:
                        head_losses.append(loss.item() * batch_size)
                        train_bbox_sample_counts[j] += batch_size
                train_epoch_head_losses_sum = [
                    epoch_sum + batch_loss
                    for epoch_sum, batch_loss in zip(
                        train_epoch_head_losses_sum, head_losses
                    )
                ]
                if self.eval_functions is not None:
                    batch_head_evals = []
                    for j, (eval_fn, head, head_output, label) in enumerate(
                        zip(self.eval_functions, self.heads, head_outputs, labels)
                    ):
                        if isinstance(head, BboxHead):
                            mask = label.sum(dim=1) != 0
                            if mask.any():
                                head_output_masked = head_output[mask]
                                label_masked = label[mask]
                                eval_value = eval_fn(head_output_masked, label_masked)
                                batch_eval = float(eval_value) * mask.sum().item()
                            else:
                                batch_eval = 0.0
                        else:
                            eval_value = eval_fn(head_output, label)
                            batch_eval = float(eval_value) * batch_size
                        batch_head_evals.append(batch_eval)
                    train_epoch_head_evals_sum = [
                        epoch_sum + batch_eval
                        for epoch_sum, batch_eval in zip(
                            train_epoch_head_evals_sum, batch_head_evals
                        )
                    ]

            print(
                f"\nEpoch:{epoch + 1}/{num_epochs}, Train Loss:{train_epoch_loss_sum / train_sample_count:.4f}"
            )
            metrics_to_log = [("train_loss", train_epoch_loss_sum / train_sample_count)]
            for i, head in enumerate(self.heads):
                if isinstance(head, BboxHead):
                    head_loss = (
                        train_epoch_head_losses_sum[i] / train_bbox_sample_counts[i]
                        if train_bbox_sample_counts[i] > 0
                        else 0.0
                    )
                else:
                    head_loss = train_epoch_head_losses_sum[i] / train_sample_count
                log_str = f"   Train {self.target_names[i]} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"train_{self.target_names[i]}_loss", head_loss))
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    if isinstance(head, BboxHead):
                        metric_value = (
                            train_epoch_head_evals_sum[i] / train_bbox_sample_counts[i]
                            if train_bbox_sample_counts[i] > 0
                            else 0.0
                        )
                    else:
                        metric_value = (
                            train_epoch_head_evals_sum[i] / train_sample_count
                        )
                    log_str += f", {metric_name}: {metric_value:.4f}"
                    metrics_to_log.append(
                        (f"train_{self.target_names[i]}_{metric_name}", metric_value)
                    )
                print(log_str)

            # Validation phase
            self.backbone.eval()
            for head in self.heads:
                head.eval()
            val_sample_count = 0
            val_epoch_loss_sum = 0
            val_epoch_head_evals_sum = [0 for _ in self.heads]
            val_epoch_head_losses_sum = [0 for _ in self.heads]
            val_bbox_sample_counts = [0 for _ in self.heads]

            with torch.no_grad():
                for i, (images, labels) in enumerate(self.val_loader):
                    if i >= 1:  # Limit to 1 batch for quick testing
                        break
                    labels = [labels] if isinstance(labels, torch.Tensor) else labels
                    labels = (
                        [labels[key] for key in labels.keys()]
                        if isinstance(labels, dict)
                        else labels
                    )
                    images = images.to(device)
                    labels = [label.to(device) for label in labels]
                    head_outputs = self._forward_pass(images)
                    losses = []
                    for head, loss_fn, head_output, label in zip(
                        self.heads, self.loss_functions, head_outputs, labels
                    ):
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
                    head_losses = []
                    for j, (head, loss) in enumerate(zip(self.heads, losses)):
                        if isinstance(head, BboxHead):
                            mask = (labels[j].sum(dim=1) != 0).float()
                            head_losses.append(loss.item() * mask.sum().item())
                            val_bbox_sample_counts[j] += mask.sum().item()
                        else:
                            head_losses.append(loss.item() * batch_size)
                            val_bbox_sample_counts[j] += batch_size
                    val_epoch_head_losses_sum = [
                        epoch_sum + batch_loss
                        for epoch_sum, batch_loss in zip(
                            val_epoch_head_losses_sum, head_losses
                        )
                    ]
                    if self.eval_functions is not None:
                        val_batch_head_evals = []
                        for j, (eval_fn, head, head_output, label) in enumerate(
                            zip(self.eval_functions, self.heads, head_outputs, labels)
                        ):
                            if isinstance(head, BboxHead):
                                mask = label.sum(dim=1) != 0
                                if mask.any():
                                    head_output_masked = head_output[mask]
                                    label_masked = label[mask]
                                    eval_value = eval_fn(
                                        head_output_masked, label_masked
                                    )
                                    batch_eval = float(eval_value) * mask.sum().item()
                                else:
                                    batch_eval = 0.0
                            else:
                                eval_value = eval_fn(head_output, label)
                                batch_eval = float(eval_value) * batch_size
                            val_batch_head_evals.append(batch_eval)
                        val_epoch_head_evals_sum = [
                            epoch_sum + batch_eval
                            for epoch_sum, batch_eval in zip(
                                val_epoch_head_evals_sum, val_batch_head_evals
                            )
                        ]

            print(
                f"Epoch:{epoch + 1}/{num_epochs}, Val Loss:{val_epoch_loss_sum / val_sample_count:.4f}"
            )
            metrics_to_log.append(("val_loss", val_epoch_loss_sum / val_sample_count))
            for i, head in enumerate(self.heads):
                if isinstance(head, BboxHead):
                    head_loss = (
                        val_epoch_head_losses_sum[i] / val_bbox_sample_counts[i]
                        if val_bbox_sample_counts[i] > 0
                        else 0.0
                    )
                else:
                    head_loss = val_epoch_head_losses_sum[i] / val_sample_count
                log_str = f"  Val {self.target_names[i]} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"val_{self.target_names[i]}_loss", head_loss))
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    if isinstance(head, BboxHead):
                        metric_value = (
                            val_epoch_head_evals_sum[i] / val_bbox_sample_counts[i]
                            if val_bbox_sample_counts[i] > 0
                            else 0.0
                        )
                    else:
                        metric_value = val_epoch_head_evals_sum[i] / val_sample_count
                    log_str += f", {metric_name}: {metric_value:.4f}"
                    metrics_to_log.append(
                        (f"val_{self.target_names[i]}_{metric_name}", metric_value)
                    )
                print(log_str)

            # Print metrics for immediate verification
            print("Metrics to log:", metrics_to_log)
            model_name = os.path.basename(self.model_path).split(".")[0]
            self.log_performance(model_name, epoch + 1, metrics_to_log)

        print("Training finished!")
        # Note: Checkpoint saving is skipped for this quick test to save time


def convert_and_get_IoU(outputs, targets) -> float:
    """Convert bounding box formats and compute IoU."""
    outputs = convertVOCBBoxFormatToAnchorFormat(outputs)
    targets = convertVOCBBoxFormatToAnchorFormat(targets)
    iou = computeBBoxIoU(outputs, targets)
    return iou


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device}")

    print("Testing models, loader and devices.")
    backbone = CNNBackbone().to(device)
    res_backbone = ResNetBackbone().to(device)
    cnn_head = BboxHead(adapter="cnn").to(device)
    res_head = BboxHead(adapter="res").to(device)
    res_class_head = ClassifierHead(adapter="res", num_classes=37).to(device)
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

    print("\n\nPreparing quick logging validation...")

    cel_fn = nn.CrossEntropyLoss()
    mse_fn = nn.MSELoss(reduction="none")
    checkpoints_dir = "checkpoints"
    NUM_SPECIES = 2
    NUM_BREEDS = 37

    # Define a single configuration for testing
    run_dicts = [
        {
            "model_path": os.path.join(checkpoints_dir, "cnn_species"),
            "heads": [ClassifierHead(NUM_SPECIES, adapter="CNN")],
            "backbone": "cnn",
            "eval_functions": [compute_accuracy],
            "eval_function_names": ["Acc"],
            "loss_functions": [cel_fn],
            "loader_targets": ["species"],
        }
    ]

    print("Starting quick logging validation with 1 configuration...\n")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = OxfordPetDataset()
    batch_size = 4  # Small batch size for quick execution
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 1  # Single epoch for quick test
    print(f"Using {device}")

    for i, run_dict in enumerate(run_dicts[:1]):  # Limit to first configuration
        adapter = "CNN" if run_dict["backbone"] == "cnn" else "Res"
        # Add a second head with num_classes=2 to test logging differentiation
        run_dict["heads"].append(ClassifierHead(num_classes=2, adapter=adapter))
        run_dict["loss_functions"].append(cel_fn)
        run_dict["eval_functions"].append(compute_accuracy)
        run_dict["eval_function_names"].append("Acc")
        run_dict["loader_targets"].append("is_animal")
        model_path = run_dict["model_path"]
        print(f"Starting test run {i + 1}, {os.path.basename(model_path)}...")
        print("Setting up trainer...")
        trainer = Trainer(
            log_dir="logs", log_file="test_logging.json"
        )  # Separate log file for testing
        trainer.set_eval_functions(
            run_dict["eval_functions"], run_dict["eval_function_names"]
        )
        train_loader, val_loader, _ = mixed_data.create_mixed_dataloaders(
            batch_size=batch_size,
            target_type=run_dict["loader_targets"],
            bg_directory="bg-20k/train",
            mixing_ratio=5,
            use_augmentation=True,
            lazy_loading=False,
        )
        trainer.set_loaders(train_loader, val_loader)
        trainer.set_loss_functions(run_dict["loss_functions"])
        trainer.set_target_names(run_dict["loader_targets"])
        backbone = CNNBackbone()
        trainer.set_model(backbone, run_dict["heads"], model_path)
        trainer.set_optimizer(learning_rate, weight_decay)
        print("Trainer set up successfully!")
        trainer.fit_sgd(device=device)
