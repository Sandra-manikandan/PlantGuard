"""
Train a Plant Disease classifier on the PlantVillage dataset using transfer learning.

Usage:
    python train.py --data_dir path/to/PlantVillage/color --model mobilenetv2
    python train.py --data_dir path/to/PlantVillage/color --model efficientnetb0 --epochs 8 --fine_tune_epochs 5

data_dir must contain one sub-folder per class, e.g.
    color/Apple___Apple_scab/*.jpg
    color/Tomato___Late_blight/*.jpg
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

IMG_SIZE = 224
SEED = 42
OUT_DIR = Path("model")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", choices=["mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=6, help="epochs with frozen base")
    p.add_argument("--fine_tune_epochs", type=int, default=4, help="epochs with top layers unfrozen")
    p.add_argument("--unfreeze_layers", type=int, default=40)
    return p.parse_args()


def load_datasets(data_dir, batch_size):
    common = dict(
        directory=data_dir,
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        seed=SEED,
        validation_split=0.2,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_full = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names

    # split the 20% hold-out into validation (10%) and test (10%)
    n_val_batches = tf.data.experimental.cardinality(val_full).numpy()
    test_ds = val_full.take(n_val_batches // 2)
    val_ds = val_full.skip(n_val_batches // 2)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(name, num_classes):
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")  # raw pixels 0-255
    x = augment(inputs)

    if name == "mobilenetv2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:  # EfficientNetB0 has its own rescaling layer built-in and expects 0-255
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )

    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=f"plant_{name}"), base


def plot_history(histories, path):
    acc = sum([h.history["accuracy"] for h in histories], [])
    val_acc = sum([h.history["val_accuracy"] for h in histories], [])
    loss = sum([h.history["loss"] for h in histories], [])
    val_loss = sum([h.history["val_loss"] for h in histories], [])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(acc, label="train"); ax[0].plot(val_acc, label="val")
    ax[0].set_title("Accuracy"); ax[0].legend()
    ax[1].plot(loss, label="train"); ax[1].plot(val_loss, label="val")
    ax[1].set_title("Loss"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probs, axis=1))
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (OUT_DIR / "classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout(); fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)


def main():
    args = get_args()
    OUT_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds, test_ds, class_names = load_datasets(args.data_dir, args.batch_size)
    print(f"{len(class_names)} classes found")
    (OUT_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2))

    model, base = build_model(args.model, len(class_names))
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        OUT_DIR / "plant_disease_model.keras", monitor="val_accuracy", save_best_only=True
    )
    early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1)

    # Phase 1: train the new classification head only
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, early, reduce])

    # Phase 2: fine-tune the top layers of the backbone with a low learning rate
    base.trainable = True
    for layer in base.layers[: -args.unfreeze_layers]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=[ckpt, early, reduce])

    plot_history([h1, h2], OUT_DIR / "training_curves.png")

    best = tf.keras.models.load_model(OUT_DIR / "plant_disease_model.keras")
    loss, acc = best.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    evaluate(best, test_ds, class_names)
    print(f"\nSaved model + class names in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
