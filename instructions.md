# instructions.pdf

# Instructions for Weakly Supervised Segmentation with CAM

This document provides detailed steps to run the code and reproduce all reported results.

## Additional Packages

**Within the comp0197-cw1-pt environment**, install the additional required libraries:

```bash
pip install matplotlib opencv-python pytorch-grad-cam
```

Alternatively, you can use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

## Project Steps

1. Data download, preparation, and pre-training classification models
2. Generating CAMs from classification models
3. Using CAMs as pseudo-labels for segmentation models
4. Applying self-training to improve results
5. Evaluation and comparison of results
6. Experiments for the Open-Ended Question
    - Experiment 1: Effect of Irrelevant Samples
    - Experiment 2: Effect of Multi-task Training
    - Experiment 3: Effect of Self-training with Augmentations

## Step 1: Data Preparation and Pre-training Classification Models

First, prepare the data and train classification models for CAM generation:

```bash
python pre_training.py
```

This script will:
- Download the Oxford Pet Dataset (if not already downloaded)
- Prepare and split the dataset (split ratio 0.7 : 0.15 : 0.15). Set `use_augmentation=`True if you want to apply data augmentations, target_type=[`"class", "species", "bbox", "segmentation"`] with the values you would like in the target to be returned in the target.
- Train various models including CNN and ResNet variants
- Train on different tasks (species classification, breed classification, bounding box regression)
- Train multi-task models with various combinations of tasks
- Save models to the `checkpoints/` directory

Note:

Pre-training with non-animal data is also supported

- With this implementation, you will need to download the background data with separate instructions:
    - `bg_directory = download_kaggle_dataset("arnaud58/landscape-pictures")`
    - use `mixed_data.create_mixed_dataloaders(*args, bg_directory = bg_directory, **kwargs)`

## Step 2: CAM Evaluation And Extraction

Now use the trained models to evaluate and generate Class Activation Maps from checkpoints stored in `checkpoints/run_x` (run_name can be configured):

```bash
python cam_evaluation.py
```

To customize parameters for this step, edit `self_training.py` and modify these variables before running:
- `run_name` - string of checkpoints sub-folder name 
- /c
- `train_mode` - Set to `True` to retrain the model or `False` to use an existing one
- `get_new_cam` - Set to `True` to generate new CAMs or `False` to load existing ones

The script:
- Instantiates 

## Step 3: Preprocessing CAM files

- Place the best CAMs file under the folder  `cam_data`. Previously we used 
`res_species_breed_bbox_50_ClassifierHead(2)_GradCAM_idx46_cams.pt`
- If the CAMs is in 256*256, run `resize_CAM.py` with the right directory to the CAM file above, and get `resized_64_species_breed_cam_mask_raw.pt`
- if the CAMs have black pixels between the boundary and the frontground (in our most recent updates, I believe this issue is saved) , run`cleanup_CAM.py` and get 
`resized_64_species_breed_cam_mask.pt`

## Step 4: Self-Training  (Alternative Open-Ended Question)

The `self_training.py` file includes the complete self-training process. To adjust parameters:

1. Open `self_training.py`
2. In `resized_data = (torch.load("cam_data/resized_64_species_breed_cam_mask_raw.pt"))`, modify the directory if the CAM file name is different
3. `Skip_first_round = True` set this to False, as we don’t have a first round model yet
4. Modify the `epochs` `BOOTSTRAP_ROUNDS` and other variables to control the number of epochs, iterations and other settings in self-training
5. Run the script:

```bash
python self_training.py
```

1. Unet Models trained in each round of each experiment will be saved under the folder `checkpoints/Bootstrap/model`. To save time for further experiments, you can set `Skip_first_round = True`, if `first_round_model.pt` is saved under `checkpoints/EVA/`after the first round of training is completed for any experiment
2. Find the best model with highest IOU score, move it to `checkpoints/EVA/` and rename it as `best_model_selftrain.pt`
    
    This process:
    1. Uses CAM as initial pseudo-labels
    2. Trains a U-Net segmentation model on these labels
    3. Predicts segmentation on unlabeled data with confidence thresholds
    4. Adds these predictions to the training set
    5. Retrains the model
    6. Repeats for the specified number of rounds
    7. Creates visualizations in the `visualizations/` directory
    

## Step 5: Evaluation of Segmentation Models

1. Train a baseline model
    1. In `baseline_training.py`, set the desired epochs like `epochs=5`.  If you want to train from scratch,  set `epochs_previous=0` . Otherwise you can loaded a previously trained baseline model and continue on training.
    2. baseline model will be saved under`checkpoints/EVA/`
2. Evaluate models
    1. In `final_evaluation_models.py`, modify the file name `model_name=f"first_round_model"` to the model that you want to evaluate under the folder`checkpoints/EVA/` 
    2. run `final_evaluation_models.py`, it will evaluate the model both on the small validation set and the testing set. 

## Step 6: Experiments for Open-Ended Questions

### Experiment 1: Effect of Irrelevant Samples

Test if adding irrelevant samples (images without pets) helps improve CAM quality:

```bash
python mixed_data.py
```

This script:
- Downloads landscape images as background/irrelevant samples
- Creates mixed datasets with pet and landscape images. Need to download the background images with the following code 
`bg_directory = download_kaggle_dataset("arnaud58/landscape-pictures")`.  Then pass 
`bg_directory=bg_directory` into the`create_mixed_dataloaders` function.
- Creates dataloaders suitable for training

### Experiment 2: Effect of Multi-task Training

Pre-training.py already generates models for multi-task training. The different model variants include:
- Single-task: cnn_species, cnn_breed, cnn_bbox, res_species, res_breed, res_bbox
- Two-task combinations: cnn_breed_species, cnn_breed_bbox, cnn_species_bbox, etc.
- Three-task combinations: cnn_species_breed_bbox, res_species_breed_bbox

To compare their performance, examine the file `pretraining.json` in the logs directory.

### Experiment 3: Effect of Self-training with Augmentations

To experiment with augmentations during self-training:

1. Open `data.py`
2. Find the `create_dataloaders` function
3. Modify the `use_augmentation` parameter to `True` when calling this function
4. Run self_training.py as before

## Directory Structure

- `oxford_pet_data/`: Dataset directory
- `checkpoints/`: Saved model checkpoints
- `checkpoints/Bootstrap/` : Saved self-training models and images (directory will be automatically created when running self-training)
- `checkpoints/EVA/` : Saved baseline models
- `cam_data/`: Generated CAM pseudo-labels
- `visualizations/`: Generated visualization outputs
- `logs/`: Training logs

## Key Files Description

- `data.py`: Oxford Pet Dataset handling
- `pre_training.py`: Pre-training classification models
- `models.py`: Model architectures (CNN, ResNet, U-Net)
- `self_training.py`: Self-training implementation
- `baseline_training.py`: Training a baseline model
- `final_evaluation_models.py`: Evaluating Basic, Best and Baseline models on the small validation set and testing set
- `evaluation.py`: Evaluation metrics and functions
- `mixed_data.py`: Mixed dataset handling with background images
- `utils.py`: Utility functions