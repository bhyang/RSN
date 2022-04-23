import glob
import os
from tqdm import tqdm

import skvideo.io
import cv2


VALIDATION_VIDEOS = ["video_files/jjk_2.mp4", "video_files/spy_family_2.mp4", "video_files/tokyo_ghoul_1.mp4"]


video_paths = glob.glob('video_files/*.mp4')
frame_counter = 0

for video_path in video_paths:
    frames = skvideo.io.vread(video_path, verbosity=1)
    prefix = os.path.join("images", "train" if video_path not in VALIDATION_VIDEOS else "val")
    for frame in tqdm(frames):
        fname = str(frame_counter).zfill(10) + ".png"
        path = os.path.join(prefix, fname)
        cv2.imwrite(path, frame)
        frame_counter += 1

print("Done")