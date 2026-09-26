import csv
from collections import Counter

with open('gesture_dataset.csv') as f:
    rows = list(csv.reader(f))

labels = [row[63] for row in rows]
counts = Counter(labels)

for gesture, count in sorted(counts.items()):
    print(f'{gesture}: {count}')

print(f'\nTotal: {len(rows)}')