
####################################
#This code corresponds to making a 5-group (train_val pair) 64*64 patchset 
####################################
import os
import json
import numpy as np
import cv2

# =========================
# PATHS
# =========================

img_dir = "/home/..../MD_OPG/OPG_Image/"
mask_dir = "/home/..../MD_OPG/OPG_Annotation/"
split_dir = "/home/..../splits/"

save_dir = "/home/..../patches"

os.makedirs(save_dir, exist_ok=True)

# =========================
# PARAMETERS
# =========================

PATCH_SIZE = 64
STRIDE = 32
NEG_RATIO = 3
FOLDS = 5

# =========================
# JSON -> MASK
# =========================

def make_mask(json_path, im_shape):

    mask = np.zeros((im_shape[0], im_shape[1]), dtype=np.uint8)
    zone = None

    with open(json_path) as f:
        data = json.load(f)

    for shape in data['shapes']:

        pts = np.array(shape["points"], dtype=np.int32)
        label = shape['label']

        if label in ["p", "o"]:
            # Both "p" and "o" are caries → same mask
            mask = cv2.fillPoly(mask, [pts], 1)

        elif label == "z":
            # rectangular dental zone
            x_min, y_min = pts[0]
            x_max, y_max = pts[1]
            zone = (int(x_min), int(y_min), int(x_max), int(y_max))

        else:
            print("Unknown label in JSON:", label)

    return mask, zone


# =========================
# PATCH EXTRACTION
# =========================

def extract_patches(image, mask, zone):

    patches = []
    labels = []

    # crop dental region
    image = image[zone[1]:zone[3], zone[0]:zone[2]]
    mask = mask[zone[1]:zone[3], zone[0]:zone[2]]

    h, w = image.shape

    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):

            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            mask_patch = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            label = 1 if np.sum(mask_patch) > 0 else 0

            patches.append(patch)
            labels.append(label)

    return patches, labels


# =========================
# PATCH EXTRACTION
# =========================

def process_images(image_list):

    all_patches = []
    all_labels = []

    for img_name in image_list:

        img_path = os.path.join(img_dir, img_name)
        
        # *** CORRECTION HERE ***
        # Change: P_***_opg_.png  -->  P_***_Ann_.json
        json_name = img_name.replace("_opg_.png", "_Ann_.json") 
        json_path = os.path.join(mask_dir, json_name)

        image = cv2.imread(img_path, 0)
        
        # Add safety check here to prevent crash if image is missing
        if image is None:
            print(f"Warning: Image not found: {img_path}. Skipping.")
            continue

        mask, zone = make_mask(json_path, image.shape)

        if zone is None:
            print("Zone missing for:", img_name)
            continue

        patches, labels = extract_patches(image, mask, zone)

        all_patches.extend(patches)
        all_labels.extend(labels)

    return np.array(all_patches), np.array(all_labels)



# =========================
# CLASS DISTRIBUTION
# =========================

def print_distribution(labels):

    n0 = np.sum(labels == 0)
    n1 = np.sum(labels == 1)

    print("Class 0:", n0)
    print("Class 1:", n1)

    if n1 > 0:
        print("Ratio:", round(n0/n1,2))


# =========================
# NEGATIVE DOWNSAMPLING
# =========================

def downsample_negatives(patches, labels):

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    n_pos = len(pos_idx)

    n_neg_keep = min(len(neg_idx), NEG_RATIO * n_pos)

    neg_keep = np.random.choice(neg_idx, n_neg_keep, replace=False)

    indices = np.concatenate([pos_idx, neg_keep])

    np.random.shuffle(indices)

    return patches[indices], labels[indices]


# =========================
# POSITIVE AUGMENTATION
# =========================

def augment_positives(patches, labels):

    new_patches = []
    new_labels = []

    for p, l in zip(patches, labels):

        new_patches.append(p)
        new_labels.append(l)

        if l == 1:

            new_patches.append(np.fliplr(p))
            new_labels.append(1)

            new_patches.append(np.flipud(p))
            new_labels.append(1)

            new_patches.append(np.rot90(p))
            new_labels.append(1)

    return np.array(new_patches), np.array(new_labels)


# =========================
# PROCESS EACH FOLD
# =========================

for fold in range(1, FOLDS + 1):

    print("\n===================")
    print("Processing Fold", fold)
    print("===================")

    fold_dir = os.path.join(save_dir, f"fold{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    train_imgs = np.load(os.path.join(split_dir, f"fold{fold}_train_images.npy"))
    val_imgs = np.load(os.path.join(split_dir, f"fold{fold}_val_images.npy"))

    # =================
    # TRAIN PATCHES
    # =================

    print("\nExtracting train patches")

    X_train, y_train = process_images(train_imgs)

    print("\nTrain distribution BEFORE balancing")

    print_distribution(y_train)

    # downsample negatives
    X_train, y_train = downsample_negatives(X_train, y_train)

    print("\nAfter negative downsampling")

    print_distribution(y_train)

    # augment positives
    X_train, y_train = augment_positives(X_train, y_train)

    print("\nAfter positive augmentation")

    print_distribution(y_train)

    # =================
    # VALIDATION PATCHES
    # =================

    print("\nExtracting validation patches")

    X_val, y_val = process_images(val_imgs)

    print("\nValidation distribution")

    print_distribution(y_val)

    # =================
    # SAVE
    # =================

    np.save(os.path.join(fold_dir, "X_train.npy"), X_train)
    np.save(os.path.join(fold_dir, "y_train.npy"), y_train)

    np.save(os.path.join(fold_dir, "X_val.npy"), X_val)
    np.save(os.path.join(fold_dir, "y_val.npy"), y_val)

    print("\nSaved fold", fold)


print("\nAll folds processed successfully")

print("\nProcessing TEST dataset...")

test_images = np.load(os.path.join(split_dir, "test_images.npy"))

print(test_images)

X_test, y_test = process_images(test_images)

print("Test patches:", X_test.shape)
print("Test labels:", y_test.shape)

unique, counts = np.unique(y_test, return_counts=True)
print("Test distribution:", dict(zip(unique, counts)))

np.save(os.path.join(save_dir, "X_test.npy"), X_test)
np.save(os.path.join(save_dir, "y_test.npy"), y_test)
