import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# Set paths
root = "IndoorActionDataset-video"
output_dir = "custom-one-frame"
sample_size = 1
os.makedirs(output_dir, exist_ok=True)

# CSV records
records = []
file_num = 0

# Loop over splits
splits = ['train', 'validation', 'test']
for split in splits:
    split_dir = os.path.join(root, split)
    if not os.path.exists(split_dir):
        continue
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    print(f"Processing split: {split}")
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(f"Processing class: {class_name}")
        for video_file in tqdm(os.listdir(class_dir)):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            video_path = os.path.join(class_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count < sample_size:
                print(f"Video {video_file} has less than {sample_size} frames, skipping.")
                cap.release()
                continue
            frame_indices = sorted(set(random.sample(range(frame_count), sample_size)))

            for idx, frame_id in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Output filename
                    base = Path(video_file).stem
                    out_name = f"{class_name}_{file_num}.jpg"
                    out_path = os.path.join(os.path.join(output_dir, split), out_name)

                    # Save frame
                    cv2.imwrite(out_path, frame)
                    file_num += 1

                    # Add to CSV
                    records.append([out_path, class_name])

            cap.release()

# Save CSV
df = pd.DataFrame(records, columns=["file_path", "class"])
print(f"Total frames extracted: {len(records)}")
df.to_csv(f"video_frames_{sample_size}.csv", index=False)
