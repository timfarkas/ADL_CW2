import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from models.heads import PretrainHead


class Pretrainer:
    """
    A trainer class for managing model training with multiple heads.

    This class handles the setup and training of a backbone network with multiple
    task-specific heads, managing data loaders, loss functions, and optimization.
    """

    backbone: nn.Module
    heads: list[PretrainHead]
    device: torch.device

    def __init__(self, device, log_dir="logs", log_file="training.json"):
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
        self.target_names = None  # Added to store target-specific names
        self.log_dir = log_dir
        self.log_file = log_file
        self.device = device

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
        """
        Set the model components and save path.

        Args:
            backbone: The backbone neural network
            heads: List of task-specific heads that process backbone features
            model_path: Path where the trained model will be saved

        Raises:
            AssertionError: If inputs have invalid types
        """
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

        print(model_path)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        self.backbone = backbone
        self.heads = heads
        self.model_path = model_path

    def checkpoint_path(self, epoch: int) -> str:
        """
        Returns the path to the checkpoint file.

        Returns:
            Path to the checkpoint file
        """
        return f"{os.path.splitext(self.model_path)[0]}_checkpoint_epoch_{epoch}.pt"

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
            "epoch": epoch,
            "backbone_state_dict": self.backbone.state_dict(),
            "heads_state_dict": [head.state_dict() for head in self.heads],
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
        }

        # Add any additional information
        if additional_info is not None:
            checkpoint.update(additional_info)

        # Create checkpoint filename with epoch number
        checkpoint_path = self.checkpoint_path(epoch)
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
        assert self.backbone is not None, (
            "Backbone must be set before loading checkpoint"
        )
        assert self.heads is not None, "Heads must be set before loading checkpoint"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        # Load backbone weights
        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])

        # Load head weights
        assert len(self.heads) == len(checkpoint["heads_state_dict"]), (
            "Number of heads doesn't match checkpoint"
        )
        for head, state_dict in zip(self.heads, checkpoint["heads_state_dict"]):
            head.load_state_dict(state_dict)

        # Load optimizer if available
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Extract and return additional info
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

        self.optimizer = optim.AdamW(
            all_params, lr=learning_rate, weight_decay=weight_decay
        )

    def set_loaders(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Set the data loaders for training and validation.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data

        Raises:
            AssertionError: If inputs have invalid types
        """
        assert isinstance(train_loader, DataLoader), (
            f"train_loader must be a DataLoader, got {type(train_loader)}"
        )
        assert isinstance(val_loader, DataLoader), (
            f"val_loader must be a DataLoader, got {type(val_loader)}"
        )

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
        assert len(eval_functions) == len(fn_names), (
            "Number of evaluation functions must match number of function names"
        )
        self.eval_functions = eval_functions
        self.eval_fn_names = fn_names

    def set_target_names(self, target_names):
        """Set target names for each head to ensure unique logging."""
        self.target_names = target_names

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

        # Create or load existing log
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

        # Add metrics to the epoch entry
        for metric_name, metric_value in metrics_list:
            log_data[model_name][epoch_key][metric_name] = metric_value

        # Save updated log
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)

    def fit_model(
        self,
        num_epochs: int,
    ) -> None:
        """
        Trains model heads.

        Args:
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for the optimizer (if not already set)
            checkpoint_interval: Number of epochs between checkpoints (0 to disable)
            device: Device to run the training on ('cuda', 'cpu', etc.)
        """
        print(f"\nTraining {self.model_path} for {num_epochs} on {self.device}....")

        assert self.optimizer is not None, "Optimizer must be set before training"
        assert self.train_loader is not None, "Train loader must be set before training"
        assert self.val_loader is not None, (
            "Validation loader must be set before training"
        )
        assert self.loss_functions is not None, (
            "Loss functions must be set before training"
        )
        assert self.target_names is not None, (
            "Target names must be set before training"
        )  # Added
        assert len(self.target_names) == len(self.heads), (
            "Target names must match number of heads"
        )  # Added

        print(self.__repr__())
        print("\n--- Training Start ---\n")

        self.backbone = self.backbone.to(self.device)
        self.heads = [head.to(self.device) for head in self.heads]

        best_epoch = 0
        crr_best_val_loss = float("inf")
        for epoch in range(num_epochs):
            # Training phase
            self.backbone.train()
            for head in self.heads:
                head.train()

            train_sample_count = 0
            train_epoch_loss_sum = 0
            train_epoch_head_evals_sum = [0 for _ in self.heads]
            train_epoch_head_losses_sum = [0 for _ in self.heads]
            train_bbox_sample_counts = [
                0 for _ in self.heads
            ]  # Track samples with bboxes for each head

            print(f"Train {epoch + 1}/{num_epochs}")
            for i, (images, labels) in enumerate(self.train_loader):
                labels = (
                    [labels] if isinstance(labels, torch.Tensor) else labels
                )  ## convert label tensor to list
                labels = (
                    [labels[key] for key in labels.keys()]
                    if isinstance(labels, dict)
                    else labels
                )  ## convert label dict to list

                if i % max(1, len(self.train_loader) // 10) == 0:
                    print("|", end="", flush=True)

                images = images.to(self.device)
                labels = [label.to(self.device) for label in labels]

                self.optimizer.zero_grad()
                head_outputs = self._forward_pass(images)

                # Compute losses with masking for BboxHead
                losses = []
                for head, loss_fn, head_output, label in zip(
                    self.heads, self.loss_functions, head_outputs, labels
                ):
                    if head.is_bbox:
                        per_sample_loss = loss_fn(head_output, label).mean(
                            dim=1
                        )  # (batch_size,)
                        mask = (
                            label.sum(dim=1) != 0
                        ).float()  # 1 if bbox exists, 0 otherwise
                        if mask.sum() > 0:
                            loss = (per_sample_loss * mask).sum() / mask.sum()
                        else:
                            loss = torch.tensor(0.0, device=self.device)
                    else:
                        loss = loss_fn(head_output, label)
                    losses.append(loss)

                total_batch_loss = sum(losses)
                total_batch_loss.backward()

                self.optimizer.step()

                ## METRICS
                batch_size = len(images)
                train_sample_count += batch_size

                ## Loss
                running_loss = total_batch_loss.item() * batch_size
                train_epoch_loss_sum += running_loss

                head_losses = []
                for j, (head, loss) in enumerate(zip(self.heads, losses)):
                    if head.is_bbox:
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

                ## Head specific eval functions (accuracy, etc)
                if self.eval_functions is not None:
                    batch_head_evals = []
                    for j, (eval_fn, head, head_output, label) in enumerate(
                        zip(self.eval_functions, self.heads, head_outputs, labels)
                    ):
                        if head.is_bbox:
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

            # Log overall loss, head-specific losses and metrics for training
            print(
                f"\nEpoch:{epoch + 1}/{num_epochs}, Train Loss:{train_epoch_loss_sum / train_sample_count:.4f}"
            )

            # Prepare metrics for logging
            metrics_to_log = [("train_loss", train_epoch_loss_sum / train_sample_count)]

            # Log individual head losses and evaluation metrics for training
            for i, head in enumerate(self.heads):
                if head.is_bbox:
                    head_loss = (
                        train_epoch_head_losses_sum[i] / train_bbox_sample_counts[i]
                        if train_bbox_sample_counts[i] > 0
                        else 0.0
                    )
                else:
                    head_loss = train_epoch_head_losses_sum[i] / train_sample_count
                log_str = f"   Train {self.target_names[i]} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"train_{self.target_names[i]}_loss", head_loss))

                # Add evaluation metric to the same log line if available
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    if head.is_bbox:
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
            val_bbox_sample_counts = [
                0 for _ in self.heads
            ]  # Track samples with bboxes for each head

            with torch.no_grad():
                for images, labels in self.val_loader:
                    labels = (
                        [labels] if isinstance(labels, torch.Tensor) else labels
                    )  ## convert label tensor to list
                    labels = (
                        [labels[key] for key in labels.keys()]
                        if isinstance(labels, dict)
                        else labels
                    )  ## convert label dict to list

                    images = images.to(self.device)
                    labels = [label.to(self.device) for label in labels]

                    head_outputs = self._forward_pass(images)

                    # Compute losses with masking for BboxHead
                    losses = []
                    for head, loss_fn, head_output, label in zip(
                        self.heads, self.loss_functions, head_outputs, labels
                    ):
                        if head.is_bbox:
                            per_sample_loss = loss_fn(head_output, label).mean(dim=1)
                            mask = (label.sum(dim=1) != 0).float()
                            if mask.sum() > 0:
                                loss = (per_sample_loss * mask).sum() / mask.sum()
                            else:
                                loss = torch.tensor(0.0, device=self.device)
                        else:
                            loss = loss_fn(head_output, label)
                        losses.append(loss)
                    total_batch_loss = sum(losses)

                    ## METRICS
                    batch_size = len(images)
                    val_sample_count += batch_size

                    ## Loss
                    val_running_loss = total_batch_loss.item() * batch_size
                    val_epoch_loss_sum += val_running_loss

                    head_losses = []
                    for j, (head, loss) in enumerate(zip(self.heads, losses)):
                        if head.is_bbox:
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

                    ## Head specific eval functions (accuracy, etc)
                    if self.eval_functions is not None:
                        val_batch_head_evals = []
                        for j, (eval_fn, head, head_output, label) in enumerate(
                            zip(self.eval_functions, self.heads, head_outputs, labels)
                        ):
                            if head.is_bbox:
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

            # Log overall loss, head-specific losses and metrics for validation
            avg_val_loss = val_epoch_loss_sum / val_sample_count
            print(f"Epoch:{epoch + 1}/{num_epochs}, Val Loss:{avg_val_loss:.4f}")
            metrics_to_log.append(("val_loss", avg_val_loss))

            # Log individual head losses and evaluation metrics for validation
            for i, head in enumerate(self.heads):
                if head.is_bbox:
                    head_loss = (
                        val_epoch_head_losses_sum[i] / val_bbox_sample_counts[i]
                        if val_bbox_sample_counts[i] > 0
                        else 0.0
                    )
                else:
                    head_loss = val_epoch_head_losses_sum[i] / val_sample_count
                log_str = f"  Val {self.target_names[i]} Loss: {head_loss:.4f}"
                metrics_to_log.append((f"val_{self.target_names[i]}_loss", head_loss))

                # Add evaluation metric to the same log line if available
                if self.eval_functions is not None and i < len(self.eval_fn_names):
                    metric_name = self.eval_fn_names[i]
                    if head.is_bbox:
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

            # Log performance metrics to JSON
            model_name = os.path.basename(self.model_path).split(".")[0]
            self.log_performance(model_name, epoch + 1, metrics_to_log)

            if avg_val_loss < crr_best_val_loss:
                best_epoch = epoch + 1
                crr_best_val_loss = avg_val_loss
                print(f"Best model so far, saving to {self.model_path}")
                self.save_checkpoint(
                    epoch=epoch + 1,
                    additional_info={
                        "train_loss": train_epoch_loss_sum / train_sample_count,
                        "val_loss": avg_val_loss,
                    },
                )

        print("\nTraining finished!")

        # Save the final model at the end of training
        self.save_checkpoint(
            epoch=num_epochs,
            additional_info={
                "train_loss": train_epoch_loss_sum / train_sample_count,
                "val_loss": avg_val_loss,
                "is_final_model": True,
            },
        )

        return best_epoch
