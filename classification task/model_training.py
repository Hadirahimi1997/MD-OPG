###################################
# train_resnet18_.py
###################################

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score, accuracy_score
import pandas as pd
from classification_models.tfkeras import Classifiers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def zscore_normalize(X):
    X = X.astype("float32")
    X = (X - np.mean(X, axis=(1, 2), keepdims=True)) / (np.std(X, axis=(1, 2), keepdims=True) + 1e-8)
    return X

def to_rgb(X):
    """Convert grayscale channels (H,W) → (H,W,3)."""
    if X.ndim == 3:
        X = np.repeat(X[..., np.newaxis], 3, axis=-1)
    elif X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)
    return X

def make_model(input_shape=(64, 64, 3)):
    ResNet18, _ = Classifiers.get("resnet18")
    base_model = ResNet18(input_shape=input_shape, weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l1(0.02))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(2, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = True

    opt = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
base_dir = "/home/..../patches/"  
save_dir = "/home/..../results/"
os.makedirs(save_dir, exist_ok=True)
model_dir = os.path.join(save_dir, "models")
os.makedirs(model_dir, exist_ok=True)

fold_metrics = []

# -------------------------------------------------------
# 5‑fold cross‑validation training
# -------------------------------------------------------
for fold in range(1, 6):
    print(f"\n========== Training Fold {fold} ==========")
    fold_dir = os.path.join(base_dir, f"fold{fold}")
    X_train = np.load(os.path.join(fold_dir, "X_train.npy"))
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    X_val = np.load(os.path.join(fold_dir, "X_val.npy"))
    y_val = np.load(os.path.join(fold_dir, "y_val.npy"))

    # Normalization and format
    X_train = to_rgb((X_train))
    X_val = to_rgb((X_val))
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)

    # Class weights
    #cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.argmax(y_train, axis=1))
    #class_weights = {0: cw[0], 1: cw[1]}
    class_weights = {0:1.0, 1:4.0}

    print("Class weights:", class_weights)

    # Model
    model = make_model((64, 64, 3))

    # Callbacks
    ckpt_path = os.path.join(model_dir, f"fold{fold}_best.h5")
    callbacks = [
        EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
        ModelCheckpoint(ckpt_path, monitor="val_auc", save_best_only=True, mode="max", verbose=1),
    ]

    # Fit
    with tf.device('/GPU:0'):
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=256,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    # ---------- Evaluate validation ----------
    
    preds_val = model.predict(X_val)

    # true labels
    y_true_val = np.argmax(y_val, axis=1)

    # probability of positive class
    y_prob_val = preds_val[:, 1]

    # decision threshold
    threshold = 0.35

    # predicted labels
    y_pred_val = (y_prob_val > threshold).astype(int)

    val_f1 = f1_score(y_true_val, y_pred_val)
    val_mcc = matthews_corrcoef(y_true_val, y_pred_val)
    val_auc = roc_auc_score(y_true_val, y_prob_val)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    val_prec = precision_score(y_true_val, y_pred_val, zero_division=0)
    val_rec = recall_score(y_true_val, y_pred_val, zero_division=0)

    fold_metrics.append({
        "fold": fold,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "val_f1": val_f1,
        "val_mcc": val_mcc,
    })

# -------------------------------------------------------
# Compute mean ± std across folds
# -------------------------------------------------------
df = pd.DataFrame(fold_metrics)
summary = df.drop(columns=["fold"]).agg(["mean","std"])
df.to_csv(os.path.join(save_dir, "fold_results.csv"), index=False)
summary.to_csv(os.path.join(save_dir, "fold_summary.csv"))
print("\nCross‑validation results:")
print(summary)

# -------------------------------------------------------
# Final Test Evaluation  (using average of 5 models)
# -------------------------------------------------------
print("\n========== Evaluating on Independent TEST set ==========")
#test_dir = os.path.join(base_dir, "test")
X_test = np.load(os.path.join(base_dir, "X_test.npy"))
y_test = np.load(os.path.join(base_dir, "y_test.npy"))

X_test = to_rgb((X_test))
y_test_cat = to_categorical(y_test, num_classes=2)

# Average predictions of 5 folds
test_preds_all = []
for fold in range(1, 6):
    model_path = os.path.join(model_dir, f"fold{fold}_best.h5")
    print(f"Loading {model_path}")
    m = tf.keras.models.load_model(model_path)
    preds = m.predict(X_test, batch_size=256)
    test_preds_all.append(preds)

test_preds = np.mean(test_preds_all, axis=0)
y_pred_test = (test_preds[:,1] > threshold).astype(int)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred_test))

test_f1 = f1_score(y_test, y_pred_test)
test_mcc = matthews_corrcoef(y_test, y_pred_test)
test_auc = roc_auc_score(y_test, test_preds[:, 1])
test_acc = accuracy_score(y_test, y_pred_test)
test_prec = precision_score(y_test, y_pred_test)
test_rec = recall_score(y_test, y_pred_test)

test_metrics = {
    "test_acc": test_acc,
    "test_auc": test_auc,
    "test_precision": test_prec,
    "test_recall": test_rec,
    "test_f1": test_f1,
    "test_mcc": test_mcc,
}
pd.DataFrame([test_metrics]).to_csv(os.path.join(save_dir, "test_results.csv"), index=False)

print("\nTest set results:")
for k, v in test_metrics.items():
    print(f"{k:15s}: {v:.4f}")
