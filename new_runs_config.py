DEFAULT_IMAGE_SIZE = (224, 224)

self_learning_experiments_config = {
    "self_learning_replace_filter": {
        "dataset_management": "replace",
        "predictions_transform": "filter",
        "seed_loss": False,
    },
    "self_learning_replace_grabcut": {
        "dataset_management": "replace",
        "predictions_transform": "grabcut",
        "seed_loss": False,
    },
    "self_learning_replace_mixlabel": {
        "dataset_management": "replace",
        "predictions_transform": "mixlabel",
        "seed_loss": False,
    },
    "self_learning_add_none": {
        "dataset_management": "add",
        "predictions_transform": None,
        "seed_loss": False,
    },
    "self_learning_add_none_seedloss": {
        "dataset_management": "add",
        "predictions_transform": None,
        "seed_loss": True,
    },
    "self_learning_add_filter": {
        "dataset_management": "add",
        "predictions_transform": "filter",
        "seed_loss": False,
    },
    "self_learning_add_grabcut": {
        "dataset_management": "add",
        "predictions_transform": "grabcut",
        "seed_loss": False,
    },
    "self_learning_add_mixlabel": {
        "dataset_management": "add",
        "predictions_transform": "mixlabel",
        "seed_loss": False,
    },
}

# TODO: if possible, create some validation
runs_config = {
    "run_1": {
        "use_augmentation": False,
        "use_mixed_data": False,
    },
    "run_2": {
        "use_augmentation": True,
        "use_mixed_data": False,
    },
    "run_3": {
        "use_augmentation": False,
        "use_mixed_data": True,
    },
    "run_4": {
        "use_augmentation": True,
        "use_mixed_data": True,
    },
}
model_names = [
    "cnn_species",
    "cnn_breed",
    "cnn_bbox",
    "cnn_breed_species",
    "cnn_breed_bbox",
    "cnn_species_bbox",
    "cnn_species_breed_bbox",
    "res18_species",
    "res50_species",
    "res18_breed",
    "res50_breed",
    "res18_bbox",
    "res50_bbox",
    "res18_breed_species",
    "res50_breed_species",
    "res18_breed_bbox",
    "res50_breed_bbox",
    "res18_species_bbox",
    "res50_species_bbox",
    "res18_species_breed_bbox",
    "res50_species_breed_bbox",
]

cam_types = ["ClassicCAM", "GradCAM"]

cam_evaluation_json = "cam_evaluation.json"

cam_dataset_folder = "cam_datasets"

visualizations_folder = "visualizations"

segmentation_output_threshold = 0.5

def get_checkpoints_and_logs_dirs(run_name: str, model_name: str):
    """
    Get the checkpoints and logs directories for a given run and model name.
    
    Args:
        run_name (str): The name of the run.
        model_name (str): The name of the model.

    Returns:
        tuple: A tuple containing the checkpoints directory and logs
        directory, in that order.
    """
    logs_dir = f"logs/{run_name}"
    checkpoints_dir = f"checkpoints/{run_name}/{model_name}"
    return checkpoints_dir, logs_dir