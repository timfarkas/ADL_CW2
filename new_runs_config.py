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