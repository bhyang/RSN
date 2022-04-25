"""
Run the network on an anime sequence and produce video output.
"""

import pdb
import os
from posixpath import splitext
GPUS = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
os.environ["WORLD_SIZE"] = str(len(GPUS.split(",")))

from unittest import loader
from tqdm import tqdm
import numpy as np
import cv2
import argparse
import json
import torch
import logging
import torch.distributed as dist

import matplotlib.pyplot as plt

from config import cfg 
from network import RSNReversalNet

from cvpack.utils.logger import get_logger
from lib.utils.dataloader import get_anime_seq_loader
from lib.utils.comm import is_main_process, synchronize, all_gather
from lib.utils.transforms import flip_back

vlogger = logging.getLogger()
old_level = vlogger.level

def get_results(outputs, kernel=11, shifts=[0.25]):
    #scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
            cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                    int(round(x) + 1e-9)]
        # aligned or not ...
        # kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
        #         centers[i][0] - scales[i][0] * 0.5
        # kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
        #         centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores 
    
    return preds, maxvals


def visualize_keypoints(image, keypoints, output_dir, idx):
    """
    draws all the joints detected on an image.
    """
    # level setting is to suppress annoying matplotlib warning
    vlogger.setLevel(100)
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    image_npy = image[0].permute(1, 2, 0).cpu().numpy()
    #image_npy = cv2.cvtColor(image_npy, cv2.COLOR_BGR2RGB)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image_npy)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for joints in keypoints:
        for j1, j2 in pairs:
            joints1 = joints[j1-1]
            joints2 = joints[j2-1]
            ax.plot([joints1[0], joints2[0]], [joints1[1], joints2[1]])
    fig.savefig(os.path.join(output_dir, "%d.png".zfill(12) % idx), bbox_inches='tight', pad_inches=0)
    plt.close()
    vlogger.setLevel(old_level)
    return

def compute_on_dataset(model, data_loader, output_dir, device):
    """
    Run inference, draw a visualization on the image, and save the image to output.

    Require batch size = 1, and unshuffled. 
    """
    model.eval()
    results = list()
    
    data = tqdm(data_loader) if is_main_process() else data_loader
    for idx, batch in enumerate(data):
        imgs = batch
        # Require batch size 1.
        assert imgs.size(0) == 1

        imgs = imgs.to(device)
        domains = torch.zeros(imgs.size(0), 1)
        with torch.no_grad():
            outputs = model(imgs, domains)
            outputs = outputs.cpu().numpy()

        #centers = np.array(centers)
        #scales = np.array(scales)
        preds, maxvals = get_results(outputs, cfg.TEST.GAUSSIAN_KERNEL, \
            cfg.TEST.SHIFT_RATIOS)

        preds = np.concatenate((preds, maxvals), axis=2)
        keypoints = []
        for i in range(preds.shape[0]):
            kpts = preds[i].reshape(17, 3)
            keypoints.append(kpts)

        # Draw keypoints. 
        visualize_keypoints(imgs, keypoints, output_dir, idx)

    return results 

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, logger):
    if is_main_process():
        logger.info("Accumulating ...")
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for p in all_predictions:
        predictions.extend(p)
    
    return predictions


def inference(model, data_loader, logger, output_path, device="cuda"):
    predictions = compute_on_dataset(model, data_loader, output_path, device)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(
            predictions, logger)

    if not is_main_process():
        return

    return predictions

def write_images_to_video(image_dir_path, video_output_path):
    image_list = sorted(os.listdir(image_dir_path), key =lambda x: int(os.path.splitext(x)[0]))
    # Read the first image and check its shape to initialize video
    # writer.
    read_img = cv2.imread(os.path.join(image_dir_path, image_list[0]))
    (height, width, _) = read_img.shape
    size = (width, height)
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for image in tqdm(image_list):
        img = cv2.imread(os.path.join(image_dir_path, image))
        video_writer.write(img)

    video_writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    parser.add_argument("--sequence", type=str, default=None)

    args = parser.parse_args()

    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if is_main_process() and not os.path.exists(cfg.TEST_DIR):
        os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.")

    model = RSNReversalNet(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(cfg.MODEL.DEVICE)

    model_file = os.path.join(cfg.OUTPUT_DIR, "iter-{}.pth".format(args.iter))
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    data_loader = get_anime_seq_loader(cfg, num_gpus, args.local_rank, 'val',
            is_dist=distributed, anime_seq=args.sequence)
    print("running inference")
    results = inference(model, data_loader, logger, cfg.VIZ_DIR, device)
    synchronize()

    if is_main_process():
        logger.info("Dumping results ...")
        results.sort(
                key=lambda res:(res['image_id'], res['score']), reverse=True) 
        results_path = os.path.join(cfg.TEST_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        logger.info("Get all results.")
        #pdb.set_trace()
        #data_loader.ori_dataset.evaluate(results_path)
    write_images_to_video(cfg.VIZ_DIR, "spy_family_test.avi")
    


if __name__ == '__main__':
    main()
