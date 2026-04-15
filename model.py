"""
Phase 3 — LSTM model training
NGT Sign Language Demonstrator project

Expects data in:
  dataset/
  ├── SIGN_ONE/
  │   ├── 0.npy
  │   └── 1.npy
  ├── SIGN_TWO/
  │   └── 0.npy
  └── ...

Each .npy file must be a (30, 225) array (output of Phase 2).

Output:
  model/
  ├── ngt_model.h5       — trained model
  └── labels.txt         — sign labels in the correct order

Usage:
  python phase3_train.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset_augmented"    # output of Phase 2
MODEL_DIR    = "model"      # where the trained model will be saved
SEQUENCE_LEN = 30           # must match Phase 2
NUM_FEATURES = 225          # must match Phase 2
TEST_SPLIT   = 0.2          # 20% of data used for testing
EPOCHS       = 200          # max training epochs (early stopping will likely stop sooner)
BATCH_SIZE   = 16

# ── Load dataset ───────────────────────────────────────────────────────────────
def load_dataset(dataset_dir):
    """
    Loads all .npy files from the dataset folder.
    Returns:
      X      — numpy array of shape (N, 30, 225)
      y      — numpy array of integer labels, shape (N,)
      labels — list of sign names in label order
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset folder '{dataset_dir}' not found. Run Phase 2 first.")

    labels = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not labels:
        raise ValueError(f"No sign folders found in '{dataset_dir}'.")

    print(f"Found {len(labels)} sign(s): {', '.join(labels)}\n")

    X, y = [], []

    for label_idx, sign in enumerate(labels):
        sign_dir = os.path.join(dataset_dir, sign)
        sequences = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]

        if not sequences:
            print(f"  [!] No .npy files found for '{sign}', skipping.")
            continue

        print(f"  {sign}: {len(sequences)} sequence(s)")

        for seq_file in sequences:
            seq_path = os.path.join(sign_dir, seq_file)
            sequence = np.load(seq_path)

            # Validate shape
            if sequence.shape != (SEQUENCE_LEN, NUM_FEATURES):
                print(f"    [!] Unexpected shape {sequence.shape} in {seq_file}, skipping.")
                continue

            X.append(sequence)
            y.append(label_idx)

    if not X:
        raise ValueError("No valid sequences found. Check your dataset folder.")

    return np.array(X), np.array(y), labels


# ── Build model ────────────────────────────────────────────────────────────────
def build_model(num_classes, sequence_len, num_features):
    """
    Balanced LSTM model:
    - Two LSTM layers to capture temporal patterns
    - Dropout layers to prevent overfitting on small datasets
    - Dense output layer with softmax for classification
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_len, num_features)),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Silence TensorFlow info messages
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    print("=" * 50)
    print("Phase 3 — LSTM Training")
    print("=" * 50 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    X, y, labels = load_dataset(DATASET_DIR)
    num_classes = len(labels)

    print(f"\nTotal sequences: {len(X)}")
    print(f"Sequence shape:  {X.shape}")
    print(f"Number of signs: {num_classes}\n")

    # Warn if very little data
    if len(X) < num_classes * 2:
        print("⚠  Warning: very few sequences per sign. Consider running data augmentation first.")
        print("   The model may not train reliably.\n")

    # ── Train/test split ───────────────────────────────────────────────────────
    # If there's only 1 sample per class, skip the split and train on everything
    if len(X) <= num_classes:
        print("Only 1 sample per sign detected — training on full dataset (no test split).\n")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=42
        )

    # Convert labels to one-hot encoded vectors
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}\n")

    # ── Build model ────────────────────────────────────────────────────────────
    print("Building model...")
    model = build_model(num_classes, SEQUENCE_LEN, NUM_FEATURES)
    model.summary()
    print()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "ngt_model.h5")

    callbacks = [
        # Stop training early if validation loss stops improving
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        # Save the best model automatically during training
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Training...\n")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Evaluation")
    print("=" * 50)

    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.1f}%")
    print(f"Test loss:     {loss:.4f}\n")

    # Per-sign breakdown
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("Per-sign results:")
    print(classification_report(y_test, y_pred, target_names=labels, zero_division=0))

    # ── Save labels ────────────────────────────────────────────────────────────
    labels_path = os.path.join(MODEL_DIR, "labels.txt")
    with open(labels_path, "w") as f:
        for label in labels:
            f.write(label + "\n")

    print(f"Model saved  → {model_path}")
    print(f"Labels saved → {labels_path}")
    print("\nPhase 3 complete. Ready for Phase 4 (live inference).")


if __name__ == "__main__":
    main()