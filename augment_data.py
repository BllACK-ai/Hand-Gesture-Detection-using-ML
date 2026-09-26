import pandas as pd
import numpy as np

df = pd.read_csv("gesture_dataset.csv", header=None)
X = df.iloc[:, :63]
y = df.iloc[:, 63]

# Classes to augment and their target count
AUGMENT_TARGETS = {
    "strafe_left":  120,
    "strafe_right": 120,
    "forward":      120,
    "land":         120,
}

augmented_rows = []

for gesture, target in AUGMENT_TARGETS.items():
    gesture_df = df[y == gesture]
    current = len(gesture_df)
    needed = target - current

    if needed <= 0:
        continue

    print(f"Augmenting {gesture}: {current} → {target}")

    for _ in range(needed):
        # Pick a random sample and add small noise
        sample = gesture_df.sample(1).values[0].copy()
        # Add gaussian noise to landmark coordinates only (not label)
        noise = np.random.normal(0, 0.005, 63)
        sample[:63] = sample[:63].astype(float) + noise
        augmented_rows.append(sample)

# Combine original + augmented
augmented_df = pd.DataFrame(augmented_rows)
final_df = pd.concat([df, augmented_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

final_df.to_csv("gesture_dataset_augmented.csv", header=False, index=False)
print(f"\nAugmented dataset saved → gesture_dataset_augmented.csv")
print(f"Original: {len(df)} | Final: {len(final_df)}")