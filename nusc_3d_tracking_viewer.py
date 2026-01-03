# ------------------------------------------------------------------------
# Copyright (c) 2023 Toyota Research Institute
# 3D MOT BEV Visualization for nuScenes
# ------------------------------------------------------------------------

import os
import json
import argparse
import importlib
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

from utils.nusc_utils.bbox import BBox
from utils.nusc_utils.visualizer2d import Visualizer2D


# =========================
# Argument Parser
# =========================

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='3D Tracking BEV Visualization')
    parser.add_argument('--config', default='utils/nusc_utils/configs/nusc_tracking_config.py', help='Config file path')
    parser.add_argument('--result', required=True, help='Tracking result json file')
    parser.add_argument('--show-dir', required=True, help='Directory to save visualizations')
    return parser.parse_args()


# =========================
# Utility Functions
# =========================

def lidar_to_global(points, sample_info):
    """
    Transform lidar points from LiDAR frame to global frame.

    Args:
        points (np.ndarray): (N, 3) lidar points
        sample_info (dict): nuScenes sample info

    Returns:
        points_global (np.ndarray): (N, 3) points in global frame
        l2g (np.ndarray): 4x4 lidar-to-global transformation matrix
    """
    l2e_r = Quaternion(sample_info['lidar2ego_rotation']).rotation_matrix
    l2e_t = sample_info['lidar2ego_translation']
    e2g_r = Quaternion(sample_info['ego2global_rotation']).rotation_matrix
    e2g_t = sample_info['ego2global_translation']

    l2e = np.eye(4)
    e2g = np.eye(4)
    l2e[:3, :3], l2e[:3, 3] = l2e_r, l2e_t
    e2g[:3, :3], e2g[:3, 3] = e2g_r, e2g_t

    pts = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts = (pts @ l2e.T) @ e2g.T

    return pts[:, :3], e2g @ l2e


def get_max_traj_length(traj_dict):
    """
    Get maximum trajectory length.
    """
    return max(len(v) for v in traj_dict.values())


def traj_dict_to_array(traj_dict):
    """
    Convert trajectory dictionary to numpy array for plotting.

    Args:
        traj_dict: {track_id: [(x, y), ...]}

    Returns:
        np.ndarray: (N_traj, T, 2)
    """
    max_len = get_max_traj_length(traj_dict)
    traj_list = []

    for traj in traj_dict.values():
        traj = [list(p) for p in traj]
        # pad trajectory with last point for alignment
        traj += [traj[-1]] * (max_len - len(traj))
        traj_list.append(traj)

    return np.array(traj_list)


# =========================
# Visualization Logic
# =========================

def visualize_one_frame(
    sample_idx,
    sample_token,
    dataset,
    sample_info,
    frame_results,
    all_results,
    save_dir
):
    """
    Visualize one frame with point cloud, bounding boxes and trajectories.
    """
    raw_data = dataset[sample_idx]

    # -------- Load & filter point cloud --------
    points = raw_data['points'].data[0].numpy()[:, :3]
    points = points[np.max(points, axis=1) < 60]  # 60m range

    points, l2g = lidar_to_global(points, sample_info)
    ego_xyz = l2g[:3, 3]

    # -------- Initialize visualizer --------
    visualizer = Visualizer2D(name=str(sample_idx), figsize=(20, 20))
    color_keys = list(visualizer.COLOR_MAP.keys())

    visualizer.handler_pc(points)

    plt.xlim((ego_xyz[0] - 60, ego_xyz[0] + 60))
    plt.ylim((ego_xyz[1] - 60, ego_xyz[1] + 60))

    # -------- Draw current frame boxes --------
    current_objects = {}   # track_id -> (x, y)
    color_list = []        # aligned with trajectory order

    for obj in frame_results:
        if obj['tracking_score'] < 0.4:
            continue

        # NOTE: assumes tracking_id format like "xxx-123"
        track_id = int(obj['tracking_id'].split('-')[-1])

        nusc_box = Box(
            obj['translation'],
            obj['size'],
            Quaternion(obj['rotation'])
        )

        bbox = BBox(
            x=nusc_box.center[0],
            y=nusc_box.center[1],
            z=nusc_box.center[2],
            w=nusc_box.wlh[0],
            l=nusc_box.wlh[1],
            h=nusc_box.wlh[2],
            o=nusc_box.orientation.yaw_pitch_roll[0]
        )

        color_key = color_keys[track_id % len(color_keys)]
        visualizer.handler_box(
            bbox,
            message=str(track_id),
            color=color_key
        )

        current_objects[track_id] = np.array(obj['translation'][:2])
        color_list.append(visualizer.COLOR_MAP[color_key])

    # -------- Collect history trajectories --------
    traj_dict = {}

    for past_token in all_results.keys():
        if past_token == sample_token:
            break

        for obj in all_results[past_token]:
            if obj['tracking_score'] < 0.4:
                continue

            track_id = int(obj['tracking_id'].split('-')[-1])
            if track_id not in current_objects:
                continue

            point = np.array(obj['translation'][:2])
            traj_dict.setdefault(track_id, []).append(point)

    # append current position as trajectory end
    for track_id, cur_point in current_objects.items():
        traj_dict.setdefault(track_id, []).append(cur_point)

    # -------- Draw trajectories --------
    if len(traj_dict) > 0:
        trajs = traj_dict_to_array(traj_dict)
        for i in range(trajs.shape[0]):
            plt.plot(trajs[i, :, 0], trajs[i, :, 1], color=color_list[i])

    # -------- Save visualization --------
    os.makedirs(save_dir, exist_ok=True)
    visualizer.save(os.path.join(save_dir, f'{sample_idx}.png'))
    visualizer.close()


# =========================
# Video Generation
# =========================

def make_video(fig_dir, fig_names, video_name):
    """
    Generate BEV video from image sequence.
    """
    import imageio
    import cv2

    writer = imageio.get_writer(
        os.path.join(fig_dir, video_name), fps=2
    )

    for name in fig_names:
        img = imageio.imread(os.path.join(fig_dir, name))
        img = cv2.resize(img, (2000, 2000))
        writer.append_data(img)

    writer.close()


# =========================
# Main Function
# =========================

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    importlib.import_module('utils.nusc_utils')

    dataset = build_dataset(cfg.data.visualization)
    data_infos = dataset.data_infos

    # map token -> index (avoid O(N) search)
    token2idx = {info['token']: i for i, info in enumerate(data_infos)}

    results = json.load(open(args.result))['results']

    pbar = tqdm(total=len(results))
    for sample_token in results.keys():
        sample_idx = token2idx[sample_token]
        sample_info = data_infos[sample_idx]

        scene_dir = os.path.join(
            args.show_dir, sample_info['scene_token']
        )

        visualize_one_frame(
            sample_idx,
            sample_token,
            dataset,
            sample_info,
            results[sample_token],
            results,
            scene_dir
        )

        pbar.update(1)
    pbar.close()

    # -------- Make videos for each scene --------
    print('Making videos...')
    for scene_token in os.listdir(args.show_dir):
        scene_dir = os.path.join(args.show_dir, scene_token)
        fig_names = sorted(
            [f for f in os.listdir(scene_dir) if f.endswith('.png')],
            key=lambda x: int(x.split('.')[0])
        )
        make_video(scene_dir, fig_names, 'videobev.mp4')


if __name__ == '__main__':
    main()
