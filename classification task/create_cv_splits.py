####################################
# This code corresponds to splitting data for "5_fold_CV" based on patient-level splitting.
####################################

import os
import numpy as np

# ==============================
# DATASET PATHS (EDIT IF NEEDED)
# ==============================

img_dir = r"/home/..../MD_OPG/OPG_Image/"
mask_dir = r"/home/..../MD_OPG/OPG_Annotation/"

save_dir = r"/home/..../splits"

# ==============================
# SETTINGS
# ==============================

TEST_SIZE = 20
N_FOLDS = 5
RANDOM_SEED = 42

# ==============================
# CREATE SAVE DIRECTORY
# ==============================

os.makedirs(save_dir, exist_ok=True)

# ==============================
# LOAD FILENAMES
# ==============================

image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".json")])

print("Images found:", len(image_files))
print("Masks found:", len(mask_files))

if len(image_files) != len(mask_files):
    raise ValueError("Number of images and masks does not match!")

n_total = len(image_files)

# ==============================
# RANDOM PERMUTATION
# ==============================

np.random.seed(RANDOM_SEED)
perm = np.random.permutation(n_total)

# ==============================
# TEST SET
# ==============================

test_indices = perm[:TEST_SIZE]
cv_indices = perm[TEST_SIZE:]

test_images = [image_files[i] for i in test_indices]
test_masks = [mask_files[i] for i in test_indices]

cv_images = [image_files[i] for i in cv_indices]
cv_masks = [mask_files[i] for i in cv_indices]

print("Test set size:", len(test_images))
print("CV pool size:", len(cv_images))

# ==============================
# SAVE TEST SET
# ==============================

np.save(os.path.join(save_dir, "test_images.npy"), np.array(test_images))
np.save(os.path.join(save_dir, "test_masks.npy"), np.array(test_masks))

print("Test set saved")

# ==============================
# CREATE FOLDS
# ==============================

def create_folds(images, masks, n_folds):

    indices = np.arange(len(images))
    np.random.shuffle(indices)

    fold_sizes = len(indices) // n_folds

    folds = []

    for fold in range(n_folds):

        start = fold * fold_sizes

        if fold == n_folds - 1:
            end = len(indices)
        else:
            end = (fold + 1) * fold_sizes

        val_idx = indices[start:end]

        train_idx = np.concatenate([indices[:start], indices[end:]])

        train_images = [images[i] for i in train_idx]
        train_masks = [masks[i] for i in train_idx]

        val_images = [images[i] for i in val_idx]
        val_masks = [masks[i] for i in val_idx]

        folds.append((train_images, train_masks, val_images, val_masks))

    return folds


folds = create_folds(cv_images, cv_masks, N_FOLDS)

# ==============================
# SAVE FOLDS
# ==============================

for i, fold in enumerate(folds):

    train_imgs, train_masks, val_imgs, val_masks = fold

    np.save(os.path.join(save_dir, f"fold{i+1}_train_images.npy"), np.array(train_imgs))
    np.save(os.path.join(save_dir, f"fold{i+1}_train_masks.npy"), np.array(train_masks))

    np.save(os.path.join(save_dir, f"fold{i+1}_val_images.npy"), np.array(val_imgs))
    np.save(os.path.join(save_dir, f"fold{i+1}_val_masks.npy"), np.array(val_masks))

    print(f"Fold {i+1} saved")
    print("Train:", len(train_imgs), " Val:", len(val_imgs))

# ==============================
# VERIFY NO OVERLAP
# ==============================

print("\nVerification:")

for i in range(N_FOLDS):

    train = np.load(os.path.join(save_dir, f"fold{i+1}_train_images.npy"))
    val = np.load(os.path.join(save_dir, f"fold{i+1}_val_images.npy"))

    overlap = set(train) & set(val)

    print(f"Fold {i+1} overlap:", len(overlap))

print("\nAll splits created successfully.")
print("Saved in:", save_dir)
