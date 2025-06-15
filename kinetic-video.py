import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

# Set paths
root = "kinetics-dataset/k400"
output_dir = "custom-kinetics"
sample_size = 1
os.makedirs(output_dir, exist_ok=True)

# CSV records
records = []
not_found = 0
file_num = 0

def find_video_path(file_paths, id):
    for file_path in file_paths:
        if id in file_path:
            return file_path
    return None

# Loop over splits
splits = ['train', 'val', 'test']
for split in splits:
    split_dir = os.path.join(root, split)
    if not os.path.exists(split_dir):
        continue
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    print(f"Processing split: {split}")
    df = pd.read_csv(os.path.join(root, "annotations", f"{split}.csv"))
    labels = df['label'].tolist()
    ids = df['youtube_id'].tolist()
    file_paths = os.listdir(split_dir)
    for id in tqdm(ids):
        video_file = find_video_path(file_paths, id)
        if not video_file: 
            not_found += 1
            #print(f"Video {id} not found in {split} split, skipping.")
            continue
        video_path = os.path.join(split_dir, video_file)
        class_name = labels[ids.index(id)]
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
print(f"Videos not found: {not_found}")
df = pd.DataFrame(records, columns=["file_path", "class"])
print(f"Total frames extracted: {len(records)}")
df.to_csv(f"kinetic_frames_{sample_size}.csv", index=False)
