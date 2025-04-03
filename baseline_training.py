import torch.nn.common_types
import baseline_model
import custom_loader
import custom_data
import torch
from baseline_model_loader import train_loader, val_loader, test_loader
import matplotlib.pyplot as plt


class TrainingBaseline:
    def initModel(self):
        segmentation_model = baseline_model.UNetWrapper(
            in_channels=3,  # 3 channels for RGB images
            n_classes=3,  # 3 class for segmentation - foreground, background, unknown
            depth=3,  # 3 encoding layers, 2 decoding layers
            wf=5,  # 2^5 = 32 channels
            padding=True,  # equivalent to padding=1 in Bruce's implementation
            batch_norm=True,  # use batch normalization after layers with activation function
            up_mode="upconv",
        )
        return segmentation_model

    ############## using simple loader for now ###############
    # def initDataset(self):
    #     # Prepare dataset
    #     dataset = custom_data.OxfordPetDataset("oxford_pet_data").prepare_dataset()

    #     # Create dataloaders
    #     train_loader, val_loader, test_loader = custom_loader.create_dataloaders(
    #         dataset,
    #         batch_size=32,
    #         target_type="segmentation",  # note no segmentation mask for some species
    #     )
    #     return train_loader, val_loader, test_loader
    ##########################################################

    def initOptimizer(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

    def train(
        self, model: torch.nn.Module, train_loader, optimizer, loss_fn, epochs=10
    ):
        # training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_train_correct = 0
            total_train_pixels = 0

            for batch_ndx, batch in enumerate(train_loader):
                images, targets = batch

                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                targets = targets.to("cuda" if torch.cuda.is_available() else "cpu")

                targets = (
                    targets.squeeze(1) - 1
                )  # Remove the channel dimension from targets

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_train_correct += (
                    (torch.argmax(outputs, dim=1) == targets).sum().item()
                )
                total_train_pixels += targets.numel()

                if batch_ndx % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_ndx}, Loss: {loss.item():.4f}"
                    )
            epoch_loss /= len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
            train_accuracy = total_train_correct / total_train_pixels
            print(f"Training Accuracy: {train_accuracy:.4f}")

            # Test the model on the validation set
            with torch.no_grad():
                val_loss = 0.0
                total_val_correct = 0
                total_val_pixels = 0
                for val_batch in val_loader:
                    val_images, val_targets = val_batch
                    val_images = val_images.to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    val_targets = val_targets.to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    val_targets = val_targets.squeeze(1) - 1
                    val_outputs = model(val_images)
                    val_loss += loss_fn(val_outputs, val_targets).item()
                    _, predicted = torch.max(val_outputs, 1)
                    total_val_correct += (predicted == val_targets).sum().item()
                    total_val_pixels += val_targets.numel()
                val_accuracy = total_val_correct / total_val_pixels
                val_loss /= len(val_loader)
                print(
                    f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {val_loss:.4f}"
                )
                print(f"Validation Loss: {val_loss:.4f}")

        print("Training complete.")
        torch.save(model.state_dict(), "baseline_model.pth")
        print("Model training complete and saved as baseline_model.pth")

    def visualisation(self, images, masks):
        # Print shape information
        print(f"Image batch shape: {images.shape}")
        print(f"Mask batch shape: {masks.shape}")
        print(f"Unique values in masks: {torch.unique(masks)}")

        # Display a few samples
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(min(4, len(images))):
            # Show image (denormalize if needed)
            img = images[i].cpu().permute(1, 2, 0)  # CHW -> HWC
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Image {i}")
            axes[0, i].axis("off")

            # Show segmentation mask
            mask = masks[i].cpu()[0]  # Remove channel dimension
            axes[1, i].imshow(mask, cmap="tab20")
            axes[1, i].set_title(f"Mask {i} - Classes: {torch.unique(mask).tolist()}")
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig("sample_batch.png")
        plt.close()
        print("Sample visualization saved to 'sample_batch.png'")

    def main(self):

        # Initialize dataset, dataloaders, optimiser and loss fn
        model = self.initModel()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = self.initOptimizer(model)
        loss = torch.nn.CrossEntropyLoss()

        # Visualise sample data
        print("Visualizing sample data...")
        sample_batch = next(iter(train_loader))
        images, masks = sample_batch
        self.visualisation(images, masks)

        # Train the model
        self.train(model, train_loader, optimizer, loss, epochs=10)
        print("Training complete.")

        # Evaluate the model
        # TODO: Implement evaluation logic


if __name__ == "__main__":
    training_baseline = TrainingBaseline()
    training_baseline.main()
