"""
AI usage statement:

AI was used to assist with researching and debugging, as well as helping
with creating docstrings. All code was writte, reviewed and/or modified by a human.
"""

from typing import Literal
import torch

from pytorch_grad_cam import AblationCAM, GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class CAMManager:
    """
    Class that manages different CAM methods based on the grad-cam package.
    It allows to visualise CAMs and can prepare outputs for self-training.
    """

    dataset: torch.utils.data.TensorDataset

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        target_type: str,
        target_layer: torch.nn.Conv2d,
        method: Literal["GradCAM", "ScoreCAM", "AblationCAM"] = "GradCAM",
        smooth: bool = False,
    ):
        """
        Args:
            model: The model to generate CAMs for
            method: CAM method ('GradCAM', 'HiResCAM', etc. naming based on grad-cam package)
            target_layer: Target layer for CAM methods, if None will be auto-detected
        """
        self.dataloader = dataloader
        self.target_type = target_type
        self.smooth = smooth
        self.model = model
        self.method = method
        self.classical = False
        self.target_layers = [target_layer]

        # Initialise the appropriate CAM method
        self.device = next(self.model.parameters()).device
        self.cam_method = method
        self.generator = self.get_cam_generator(
            dataloader=dataloader,
            target_type=target_type,
            smooth=smooth,
        )

    def get_cam_dataset(self, num_samples=None) -> torch.utils.data.TensorDataset:
        self.dataset = self._generate_cam_dataset(
            self.dataloader,
            self.target_type,
            self.smooth,
            num_samples,
        )
        return self.dataset

    def get_cam_generator(self, dataloader, target_type: str, smooth: bool = False):
        for batch_images, batch_targets in dataloader:
            cam = None
            match self.method:
                case "ClassicCAM":
                    cam = self._classic_cam(model=self.model)
                case "GradCAM":
                    cam = GradCAM(model=self.model, target_layers=self.target_layers)
                case "ScoreCAM":
                    cam = ScoreCAM(model=self.model, target_layers=self.target_layers)
                case "AblationCAM":
                    cam = AblationCAM(
                        model=self.model, target_layers=self.target_layers
                    )
                case _:
                    raise ValueError(f"Unsupported CAM method: {self.method}")
            images = batch_images.to(self.device)
            try:
                labels = batch_targets[target_type].to(self.device)
                gt_masks = batch_targets["segmentation"].to(self.device)
            except (ValueError, KeyError):
                raise ValueError(
                    f"Expected dict with keys '{target_type}' and 'segmentation'"
                )

            targets = [ClassifierOutputTarget(label.item()) for label in labels]

            grayscale_cams = cam(
                input_tensor=images,
                targets=targets,
                aug_smooth=smooth,
                eigen_smooth=smooth,
            )

            # LLM recommendation: remove hooks from cam
            if hasattr(cam, "activations_and_grads") and hasattr(
                cam.activations_and_grads, "release"
            ):
                cam.activations_and_grads.release()

            if self.method != "ClassicCAM":
                tensor_cams = (
                    torch.from_numpy(grayscale_cams)
                    .float()
                    .unsqueeze(1)
                    .to(self.device)
                )
            else:
                tensor_cams = grayscale_cams

            # Convert gt_masks to float before downsampling to avoid the Long tensor error
            gt_masks_float = gt_masks.float()
            yield (images, tensor_cams, gt_masks_float)

    def _generate_cam_dataset(
        self,
        dataloader,
        target_type,
        smooth: bool,
        num_samples=None,
    ):
        """
        Generate a dataset with CAM masks for self-training.

        Args:
            dataloader: DataLoader with images

        Returns:
            TensorDataset: Contains (images, cam_mask, segmentation_masks),
                where masks are [B, 1, H, W].
        """

        self.model.eval()

        all_images, all_cams, all_masks = [], [], []
        batch_size = dataloader.batch_size
        for i, (images, cams, masks) in enumerate(
            self.get_cam_generator(dataloader, target_type, smooth)
        ):
            all_images.append(images.detach())
            all_cams.append(cams.detach())
            all_masks.append(masks.detach())
            if num_samples is not None and (i + 1) * batch_size >= num_samples:
                break

        # Concatenate all batches
        images_tensor = torch.cat(all_images, dim=0)
        cams_tensor = torch.cat(all_cams, dim=0)
        masks_tensor = torch.cat(all_masks, dim=0)

        return torch.utils.data.TensorDataset(images_tensor, cams_tensor, masks_tensor)

    def _classic_cam(self, model: torch.nn.Module):
        def _forward(
            input_tensor,
            targets,
            aug_smooth,
            eigen_smooth,
        ):
            """
            Forward pass to get logits and feature maps.
            """
            model.eval()
            batch_cams = []
            input_size = input_tensor.shape[-2:]  # get (H, W) from input

            with torch.no_grad():
                logits = model(input_tensor)  # logits: (B, C)
                feature_maps = model.feature_maps  # (B, C, H, W)
                weights = model.head.head[2].weight
                pred_classes = logits.argmax(dim=1)  # (B,)

                for i in range(input_tensor.size(0)):
                    fmap = feature_maps[i]  # (C, H, W)
                    cls_idx = pred_classes[i]
                    weight_vec = weights[cls_idx].view(-1, 1, 1)  # (C, 1, 1)

                    cam = torch.sum(fmap * weight_vec, dim=0, keepdim=True).unsqueeze(
                        0
                    )  # (1, 1, H, W)
                    cam = torch.nn.functional.relu(cam)
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-8)
                    cam_resized = torch.nn.functional.interpolate(
                        cam, size=input_size, mode="bilinear", align_corners=False
                    )
                    batch_cams.append(cam_resized)

                batch_cams = torch.cat(batch_cams, dim=0)  # (B, 1, H, W)
            return batch_cams

        return _forward
