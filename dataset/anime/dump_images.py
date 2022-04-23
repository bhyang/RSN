import glob
import os
from tqdm import tqdm

import skvideo.io
import cv2

# Sampling frequency( ~10FPS)
SKIP_EVERY  = 3
VALIDATION_VIDEOS = ["video_files/jjk_2.mp4", "video_files/spy_family_2.mp4", "video_files/tokyo_ghoul_1.mp4"]


video_paths = glob.glob('video_files/*.mp4')
frame_counter = 0

for video_path in video_paths:
    print("Reading %s into memory ... " % video_path)
    frames = skvideo.io.vread(video_path, verbosity=0)
    prefix = os.path.join("images", "train" if video_path not in VALIDATION_VIDEOS else "val")
    print("Writing %s  to %s ..." % (video_path, prefix))
    for idx in tqdm(range(0, len(frames), SKIP_EVERY)):
        fname = str(frame_counter).zfill(10) + ".png"
        path = os.path.join(prefix, fname)
        if not os.path.exists(path):
            cv2.imwrite(path, frames[idx])
        frame_counter += 1

print("Done")