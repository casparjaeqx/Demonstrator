"""
Data Augmentation
NGT Sign Language Demonstrator project

Takes the extracted keypoint sequences from Phase 2 and generates
additional training samples by applying variations to each sequence.

Augmentation techniques applied:
  - Mirroring       — flips x coordinates (simulates opposite hand)
  - Speed up        — samples fewer frames (faster signing)
  - Slow down       — samples more frames (slower signing)
  - Noise           — adds small random shifts to all keypoints
  - Scaling         — makes keypoints larger or smaller
  - Time shift      — shifts the sequence slightly forward or backward
  - Rotation        — rotates keypoints slightly around the centre

Input:
  dataset/          (output of Phase 2, original sequences)

Output:
  dataset_augmented/  (original + augmented sequences)

Usage:
  python augment.py
"""

import os
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_DIR    = "dataset"
OUTPUT_DIR   = "dataset_augmented"
SEQUENCE_LEN = 30
NUM_FEATURES = 225   # 75 landmarks × 3 (x, y, z)

# How many augmented copies to generate per original sequence
AUGMENTS_PER_SEQUENCE = 10

# Augmentation parameters
NOISE_STD       = 0.005   # std of gaussian noise added to keypoints
SCALE_RANGE     = (0.85, 1.15)  # scale factor range
SPEED_RANGE     = (0.7, 1.3)    # speed factor range (< 1 = faster, > 1 = slower)
TIME_SHIFT_MAX  = 5       # max frames to shift forward/backward
ROTATION_RANGE  = (-15, 15)     # degrees


# ── Augmentation functions ─────────────────────────────────────────────────────

def mirror(sequence):
    """
    Flips all x coordinates around the centre (0.5).
    Also swaps left and right hand keypoints so the skeleton stays correct.
    Sequence shape: (30, 225) — 33 pose × 3 + 21 lh × 3 + 21 rh × 3
    """
    seq = sequence.copy()

    # Flip all x values (every 3rd value starting at 0)
    seq[:, 0::3] = 1.0 - seq[:, 0::3]

    # Swap left and right hand blocks
    # Pose: indices 0..98, Left hand: 99..161, Right hand: 162..224
    lh = seq[:, 99:162].copy()
    rh = seq[:, 162:225].copy()
    seq[:, 99:162]  = rh
    seq[:, 162:225] = lh

    return seq


def add_noise(sequence, std=NOISE_STD):
    """Adds small gaussian noise to all keypoints."""
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise


def scale(sequence, factor=None):
    """
    Scales keypoints around their centre point.
    Makes the signer appear closer or further from the camera.
    """
    if factor is None:
        factor = np.random.uniform(*SCALE_RANGE)

    seq = sequence.copy()
    # Scale x and y (not z) around centre 0.5
    for axis in [0, 1]:  # 0=x, 1=y
        seq[:, axis::3] = (seq[:, axis::3] - 0.5) * factor + 0.5
    return seq


def change_speed(sequence, factor=None):
    """
    Resamples the sequence at a different speed.
    factor < 1 = faster signing (fewer source frames used)
    factor > 1 = slower signing (more source frames stretched)
    """
    if factor is None:
        factor = np.random.uniform(*SPEED_RANGE)

    n = SEQUENCE_LEN
    # Source indices to sample from
    src_len     = int(n * factor)
    src_len     = max(2, min(src_len, n * 2))
    src_indices = np.linspace(0, src_len - 1, n)

    # Wrap source indices to available frames
    src_indices = np.clip(src_indices, 0, n - 1).astype(int)
    return sequence[src_indices]


def time_shift(sequence, max_shift=TIME_SHIFT_MAX):
    """
    Shifts the sequence forward or backward in time.
    Pads with zeros or truncates as needed.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return sequence.copy()

    seq = np.zeros_like(sequence)
    if shift > 0:
        seq[shift:] = sequence[:-shift]
    else:
        seq[:shift] = sequence[-shift:]
    return seq


def rotate_2d(sequence, angle_deg=None):
    """
    Rotates x,y keypoints around the centre (0.5, 0.5).
    Simulates the signer being filmed at a slightly different angle.
    """
    if angle_deg is None:
        angle_deg = np.random.uniform(*ROTATION_RANGE)

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    seq = sequence.copy()

    # Rotate every (x, y) pair
    x_indices = np.arange(0, NUM_FEATURES, 3)
    y_indices = np.arange(1, NUM_FEATURES, 3)

    x = seq[:, x_indices] - 0.5
    y = seq[:, y_indices] - 0.5

    seq[:, x_indices] = x * cos_a - y * sin_a + 0.5
    seq[:, y_indices] = x * sin_a + y * cos_a + 0.5

    return seq


# ── Generate augmented sequences ──────────────────────────────────────────────
def augment_sequence(sequence):
    """
    Generates AUGMENTS_PER_SEQUENCE variations of a single sequence
    by randomly combining augmentation techniques.
    """
    augmented = []

    for _ in range(AUGMENTS_PER_SEQUENCE):
        seq = sequence.copy()

        # Always apply a random combination of techniques
        if np.random.random() > 0.3:
            seq = add_noise(seq)
        if np.random.random() > 0.3:
            seq = scale(seq)
        if np.random.random() > 0.3:
            seq = change_speed(seq)
        if np.random.random() > 0.3:
            seq = time_shift(seq)
        if np.random.random() > 0.3:
            seq = rotate_2d(seq)

        # Always generate one mirrored copy
        augmented.append(seq)

    # Add a clean mirror of the original as a bonus sample
    augmented.append(mirror(sequence))

    return augmented


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("Data Augmentation")
    print("=" * 50 + "\n")

    if not os.path.exists(INPUT_DIR):
        print(f"Error: '{INPUT_DIR}' folder not found. Run phase2_extract.py first.")
        return

    signs = sorted([
        d for d in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, d))
    ])

    if not signs:
        print(f"No sign folders found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(signs)} sign(s): {', '.join(signs)}\n")

    total_original  = 0
    total_augmented = 0

    for sign in signs:
        input_sign_dir  = os.path.join(INPUT_DIR, sign)
        output_sign_dir = os.path.join(OUTPUT_DIR, sign)
        os.makedirs(output_sign_dir, exist_ok=True)

        sequences = [f for f in os.listdir(input_sign_dir) if f.endswith(".npy")]

        if not sequences:
            print(f"[{sign}] No sequences found, skipping.")
            continue

        print(f"[{sign}] {len(sequences)} original sequence(s)...")

        out_idx = 0

        for seq_file in sequences:
            seq_path = os.path.join(input_sign_dir, seq_file)
            sequence = np.load(seq_path)

            if sequence.shape != (SEQUENCE_LEN, NUM_FEATURES):
                print(f"  [!] Unexpected shape {sequence.shape} in {seq_file}, skipping.")
                continue

            # Copy original to output
            np.save(os.path.join(output_sign_dir, f"{out_idx}.npy"), sequence)
            out_idx += 1
            total_original += 1

            # Generate augmented versions
            augmented = augment_sequence(sequence)
            for aug_seq in augmented:
                np.save(os.path.join(output_sign_dir, f"{out_idx}.npy"), aug_seq)
                out_idx += 1
                total_augmented += 1

        print(f"  → {out_idx} total sequences saved (1 original + {out_idx - len(sequences)} augmented per source)")

    print(f"\nDone.")
    print(f"  Original sequences:  {total_original}")
    print(f"  Augmented sequences: {total_augmented}")
    print(f"  Total:               {total_original + total_augmented}")
    print(f"\nAugmented dataset saved to '{OUTPUT_DIR}/'")
    print("Point phase3_train.py at this folder by changing DATASET_DIR = \"dataset_augmented\"")


if __name__ == "__main__":
    main()