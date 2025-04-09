from data import OxfordPetDataset
import os
from models import ResNetBackbone, CNNBackbone, BboxHead
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from utils import computeBBoxIoU, convertVOCBBoxFormatToAnchorFormat
from torchvision import transforms
import json

class Trainer:
    def __init__(self, log_dir="logs", log_file="training.json"):
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

    def set_model(self, backbone, heads, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.backbone = backbone
        self.heads = heads
        self.model_path = model_path

    def set_loss_functions(self, loss_functions):
        self.loss_functions = loss_functions

    def set_optimizer(self, learning_rate, weight_decay):
        all_params = list(self.backbone.parameters())
        for head in self.heads:
            all_params.extend(head.parameters())
        self.optimizer = optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)

    def set_loaders(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_eval_functions(self, eval_functions, fn_names):
        assert len(eval_functions) == len(fn_names), "Eval functions and names must match in length"
        self.eval_functions = eval_functions
        self.eval_fn_names = fn_names

    def _forward_pass(self, x):
        features = self.backbone(x)
        return [head(features) for head in self.heads]

    def log_performance(self, model_name, epoch, metrics_list):
        os.makedirs(self.log_dir, exist_ok=True)
        log_file = os.path.join(self.log_dir, self.log_file)
        log_data = json.load(open(log_file, 'r')) if os.path.exists(log_file) else {}
        if model_name not in log_data:
            log_data[model_name] = {}
        log_data[model_name][str(epoch)] = {name: value for name, value in metrics_list}
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)

    def fit_sgd(self, num_epochs=1, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Validating {self.model_path} on {device}...")

        self.backbone.to(device)
        self.heads = [head.to(device) for head in self.heads]

        for epoch in range(num_epochs):
            self.backbone.train()
            for head in self.heads:
                head.train()

            train_sample_count = 0
            train_epoch_loss_sum = 0
            train_epoch_head_evals_sum = [0 for _ in self.heads]
            train_bbox_sample_counts = [0 for _ in self.heads]

            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = [labels["bbox"].to(device)] if isinstance(labels, dict) else [labels.to(device)]

                self.optimizer.zero_grad()
                head_outputs = self._forward_pass(images)

                for head, loss_fn, head_output, label in zip(self.heads, self.loss_functions, head_outputs, labels):
                    per_sample_loss = loss_fn(head_output, label).mean(dim=1)
                    mask = (label.sum(dim=1) != 0)  # Boolean mask
                    if mask.any():
                        loss = (per_sample_loss * mask.float()).sum() / mask.sum()
                    else:
                        loss = torch.tensor(0.0, device=device)
                    if i < 5:  # Log only first 5 batches
                        print(f"Batch {i}: Bbox loss: {loss.item():.4f}, Samples with bbox: {int(mask.sum())}/{len(images)}")
                    loss.backward()

                self.optimizer.step()

                batch_size = len(images)
                train_sample_count += batch_size
                train_epoch_loss_sum += loss.item() * batch_size
                train_bbox_sample_counts[0] += mask.sum().item()

                if self.eval_functions:
                    for j, (eval_fn, head_output, label) in enumerate(zip(self.eval_functions, head_outputs, labels)):
                        mask = (label.sum(dim=1) != 0)  # Boolean mask
                        if mask.any():
                            head_output_masked = head_output[mask]
                            label_masked = label[mask]
                            eval_value = eval_fn(head_output_masked, label_masked)
                            train_epoch_head_evals_sum[j] += float(eval_value) * mask.sum().item()
                            if i < 5:
                                print(f"Batch {i}: IoU on {mask.sum().item()} samples: {eval_value:.4f}")

            print(f"Epoch {epoch+1}: Train Loss: {train_epoch_loss_sum / train_sample_count:.4f}")
            if self.eval_functions:
                iou = train_epoch_head_evals_sum[0] / train_bbox_sample_counts[0] if train_bbox_sample_counts[0] > 0 else 0.0
                print(f"Train Bbox IoU: {iou:.4f}")

            self.backbone.eval()
            for head in self.heads:
                head.eval()

            val_sample_count = 0
            val_epoch_loss_sum = 0
            val_epoch_head_evals_sum = [0 for _ in self.heads]
            val_bbox_sample_counts = [0 for _ in self.heads]

            with torch.no_grad():
                for i, (images, labels) in enumerate(self.val_loader):
                    images = images.to(device)
                    labels = [labels["bbox"].to(device)] if isinstance(labels, dict) else [labels.to(device)]
                    head_outputs = self._forward_pass(images)

                    for head, loss_fn, head_output, label in zip(self.heads, self.loss_functions, head_outputs, labels):
                        per_sample_loss = loss_fn(head_output, label).mean(dim=1)
                        mask = (label.sum(dim=1) != 0)  # Boolean mask
                        if mask.any():
                            loss = (per_sample_loss * mask.float()).sum() / mask.sum()
                        else:
                            loss = torch.tensor(0.0, device=device)

                        batch_size = len(images)
                        val_sample_count += batch_size
                        val_epoch_loss_sum += loss.item() * batch_size
                        val_bbox_sample_counts[0] += mask.sum().item()

                        if self.eval_functions:
                            if mask.any():
                                head_output_masked = head_output[mask]
                                label_masked = label[mask]
                                eval_value = self.eval_functions[0](head_output_masked, label_masked)
                                val_epoch_head_evals_sum[0] += float(eval_value) * mask.sum().item()

            print(f"Epoch {epoch+1}: Val Loss: {val_epoch_loss_sum / val_sample_count:.4f}")
            if self.eval_functions:
                iou = val_epoch_head_evals_sum[0] / val_bbox_sample_counts[0] if val_bbox_sample_counts[0] > 0 else 0.0
                print(f"Val Bbox IoU: {iou:.4f}")

            model_name = os.path.basename(self.model_path)
            metrics = [
                ("train_loss", train_epoch_loss_sum / train_sample_count),
                ("val_loss", val_epoch_loss_sum / val_sample_count),
                ("train_bbox_iou", train_epoch_head_evals_sum[0] / train_bbox_sample_counts[0] if train_bbox_sample_counts[0] > 0 else 0.0),
                ("val_bbox_iou", val_epoch_head_evals_sum[0] / val_bbox_sample_counts[0] if val_bbox_sample_counts[0] > 0 else 0.0)
            ]
            self.log_performance(model_name, epoch + 1, metrics)

def convert_and_get_IoU(outputs, targets):
    outputs = convertVOCBBoxFormatToAnchorFormat(outputs)
    targets = convertVOCBBoxFormatToAnchorFormat(targets)
    return computeBBoxIoU(outputs, targets)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    run_dicts = [
        {"model_path": "checkpoints/cnn_bbox", "heads": [BboxHead(adapter="CNN")], "backbone": "cnn"},
        {"model_path": "checkpoints/res_bbox", "heads": [BboxHead(adapter="Res")], "backbone": "res"}
    ]

    mse_fn = nn.MSELoss(reduction='none')
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 1e-4

    for run_dict in run_dicts:
        print(f"Validating {os.path.basename(run_dict['model_path'])}...")
        trainer = Trainer()

        train_dataset = OxfordPetDataset(target_type=["bbox"], split="train", transform=transform)
        train_subset = Subset(train_dataset, range(50))  # Only 50 samples
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        val_dataset = OxfordPetDataset(target_type=["bbox"], split="val", transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        trainer.set_loaders(train_loader, val_loader)
        trainer.set_loss_functions([mse_fn])
        trainer.set_eval_functions([convert_and_get_IoU], ["IoU"])

        backbone = CNNBackbone() if run_dict["backbone"] == "cnn" else ResNetBackbone()
        trainer.set_model(backbone, run_dict["heads"], run_dict["model_path"])
        trainer.set_optimizer(learning_rate, weight_decay)
        trainer.fit_sgd(device=device)