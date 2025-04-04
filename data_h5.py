import os
import requests
import tarfile
import shutil
import random

from PIL import Image
import numpy as np
import h5py

'''This file based is based on and modified from the tutorial img_sgm'''

# ====== Configuration ======
filenames = ['images.tar.gz', 'annotations.tar.gz']
url_base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/'
DATA_PATH = 'data'
im_size = (64, 64)
ratio_val = 0.15
ratio_test = 0.15
# ============================

# Clean or create data directory
if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

# ====== Download and Extract ======
print('Downloading and extracting data...')
for fname in filenames:
    tar_path = os.path.join(DATA_PATH, fname)
    extract_needed = False

    # Download if not already saved
    if not os.path.exists(tar_path):
        url = url_base + fname
        print(f'Downloading {url}...')
        r = requests.get(url, allow_redirects=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
        extract_needed = True
    else:
        print(f'{fname} already exists. Skipping download.')
        extract_needed = True  # Always extract to be safe unless already extracted

    # Extract only if the directory isn't already there
    if extract_needed:
        with tarfile.open(tar_path) as tar:
            tar.extractall()

img_dir = 'images'
seg_dir = 'annotations/trimaps'

# ====== Build species_map from trainval.txt ======
species_map = {}  # image_id (lowercase) → (species_str, breed_str)
with open('annotations/trainval.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        image_id = parts[0]  # e.g., 'Abyssinian_100'
        breed = '_'.join(image_id.split('_')[:-1])
        species = int(parts[2])  # 1=cat, 2=dog
        species_map[image_id.lower()] = ('cat' if species == 1 else 'dog', breed)

# ====== Create Label Mappings ======
species_to_id = {'cat': 0, 'dog': 1}
breed_list = sorted(set(b for _, b in species_map.values()))
breed_to_id = {breed: idx for idx, breed in enumerate(breed_list)}

# ====== Prepare HDF5 Files ======
img_h5s, seg_h5s = [], []
for split in ["train", "val", "test"]:
    img_h5s.append(h5py.File(os.path.join(DATA_PATH, f"images_{split}.h5"), "w"))
    seg_h5s.append(h5py.File(os.path.join(DATA_PATH, f"labels_{split}.h5"), "w"))

# ====== Prepare Filenames and Split ======
img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.seed(90)
random.shuffle(img_filenames)

num_data = len(img_filenames)
num_val = int(num_data * ratio_val)
num_test = int(num_data * ratio_test)
num_train = num_data - num_val - num_test

print(f"Splitting data: {num_train} train, {num_val} val, {num_test} test")

# ====== Write HDF5 Data ======
counter = [0, 0, 0]  # track idx for each split
skipped = 0

for idx, im_file in enumerate(img_filenames):
    base_name = os.path.splitext(im_file)[0].lower()

    # Skip if metadata is missing
    if base_name not in species_map:
        print(f"Skipping {im_file}: no metadata")
        skipped += 1
        continue

    # Determine split
    if idx < num_train:
        ids = 0  # train
    elif idx < num_train + num_val:
        ids = 1  # val
    else:
        ids = 2  # test

    species, breed = species_map[base_name]
    species_id = species_to_id[species]
    breed_id = breed_to_id[breed]

    # Load and save image
    with Image.open(os.path.join(img_dir, im_file)) as img:
        img_arr = np.array(img.convert('RGB').resize(im_size), dtype='uint8')
        dset = img_h5s[ids].create_dataset(f"{counter[ids]:06d}", data=img_arr)
        dset.attrs['species'] = species_id
        dset.attrs['breed'] = breed_id

    # Load and save segmentation mask
    with Image.open(os.path.join(seg_dir, base_name + '.png')) as seg:
        seg_arr = np.array(seg.resize(im_size), dtype='uint8')
        seg_h5s[ids].create_dataset(f"{counter[ids]:06d}", data=seg_arr)

    counter[ids] += 1

# ====== Finalize ======
for h5 in img_h5s + seg_h5s:
    h5.flush()
    h5.close()

shutil.rmtree(img_dir)
shutil.rmtree(seg_dir.split('/')[0])  # remove annotations/

print(f'Dataset saved to {os.path.abspath(DATA_PATH)}')
print(f"Total skipped due to missing metadata: {skipped}")
