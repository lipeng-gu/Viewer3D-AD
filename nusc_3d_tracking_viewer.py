# ------------------------------------------------------------------------
# 3D MOT BEV Visualization (Prediction + GT)
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


# ===============================
# Argument
# ===============================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='utils/nusc_utils/configs/nusc_tracking_config.py', help='Config file path')
    parser.add_argument('--show_dir', default='work_dirs/vis_results_nusc', help='output directory')
    parser.add_argument('--score_thresh', default=0.2, help='output directory')
    parser.add_argument('--result', required=True, help='tracking result json')
    return parser.parse_args()


# ===============================
# Geometry utils
# ===============================

def lidar_to_global(points, info):
    l2e_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    l2e_t = info['lidar2ego_translation']
    e2g_r = Quaternion(info['ego2global_rotation']).rotation_matrix
    e2g_t = info['ego2global_translation']

    l2e = np.eye(4)
    e2g = np.eye(4)
    l2e[:3, :3], l2e[:3, 3] = l2e_r, l2e_t
    e2g[:3, :3], e2g[:3, 3] = e2g_r, e2g_t

    pts = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    pts = (pts @ l2e.T) @ e2g.T

    return pts[:, :3], e2g @ l2e


# ===============================
# Trajectory utils
# ===============================

def traj_dict_to_array(traj_dict):
    max_len = max(len(v) for v in traj_dict.values())
    trajs = []

    for t in traj_dict.values():
        t = [list(p) for p in t]
        t += [t[-1]] * (max_len - len(t))
        trajs.append(t)

    return np.array(trajs)


# ===============================
# GT utils
# ===============================

def build_gt_bbox(box, l2g):
    bbox = BBox(
        x=box[0], y=box[1], z=box[2],
        w=box[3], l=box[4], h=box[5],
        o=-(box[6] + np.pi / 2)
    )
    return BBox.bbox2world(l2g, bbox)


def collect_gt_traj(sample_idx, data_infos, instance_ids):
    traj = {}
    for i in range(sample_idx + 1):
        info = data_infos[i]
        for box, ins_id in zip(info['gt_boxes'], info['instance_inds']):
            if ins_id in instance_ids:
                traj.setdefault(ins_id, []).append(box[:2])
    return traj


# ===============================
# Visualization
# ===============================

def visualize_pred(
    sample_idx, sample_token, dataset, info,
    frame_results, all_results, score_thresh, save_dir
):
    raw = dataset[sample_idx]
    points = raw['points'].data[0].numpy()[:, :3]
    points = points[np.max(points, axis=1) < 60]

    points, l2g = lidar_to_global(points, info)
    ego = l2g[:3, 3]

    vis = Visualizer2D(f'pred_{sample_idx}', figsize=(20, 20))
    colors = list(vis.COLOR_MAP.keys())
    vis.handler_pc(points)

    plt.xlim(ego[0]-60, ego[0]+60)
    plt.ylim(ego[1]-60, ego[1]+60)

    current = {}
    color_list = []

    for obj in frame_results:
        if obj['tracking_score'] < score_thresh:
            continue

        tid = int(obj['tracking_id'])
        box = Box(obj['translation'], obj['size'], Quaternion(obj['rotation']))

        bbox = BBox(
            x=box.center[0], y=box.center[1], z=box.center[2],
            w=box.wlh[0], l=box.wlh[1], h=box.wlh[2],
            o=box.orientation.yaw_pitch_roll[0]
        )

        ck = colors[tid % len(colors)]
        vis.handler_box(bbox, message=str(tid), color=ck)
        current[tid] = np.array(obj['translation'][:2])
        color_list.append(vis.COLOR_MAP[ck])

    traj = {}
    for tk in all_results:
        if tk == sample_token:
            break
        for obj in all_results[tk]:
            if obj['tracking_score'] < score_thresh:
                continue
            tid = int(obj['tracking_id'])
            if tid in current:
                traj.setdefault(tid, []).append(obj['translation'][:2])

    for tid, pt in current.items():
        traj.setdefault(tid, []).append(pt)

    if traj:
        trajs = traj_dict_to_array(traj)
        for i in range(trajs.shape[0]):
            plt.plot(trajs[i,:,0], trajs[i,:,1], color=color_list[i])

    os.makedirs(save_dir, exist_ok=True)
    vis.save(os.path.join(save_dir, f'{sample_idx}.png'))
    vis.close()


def visualize_gt(
    sample_idx, dataset, info,
    data_infos, save_dir
):
    raw = dataset[sample_idx]
    points = raw['points'].data[0].numpy()[:, :3]
    points = points[np.max(points, axis=1) < 60]

    points, l2g = lidar_to_global(points, info)
    ego = l2g[:3, 3]

    vis = Visualizer2D(f'gt_{sample_idx}', figsize=(20, 20))
    colors = list(vis.COLOR_MAP.keys())
    vis.handler_pc(points)

    plt.xlim(ego[0]-60, ego[0]+60)
    plt.ylim(ego[1]-60, ego[1]+60)

    ids = []
    color_list = []

    for box, ins_id in zip(info['gt_boxes'], info['instance_inds']):
        bbox = build_gt_bbox(box, l2g)
        ck = colors[ins_id % len(colors)]
        vis.handler_box(bbox, message=str(ins_id), color=ck)
        ids.append(ins_id)
        color_list.append(vis.COLOR_MAP[ck])

    traj = collect_gt_traj(sample_idx, data_infos, ids)

    if traj:
        trajs = traj_dict_to_array(traj)
        for i in range(trajs.shape[0]):
            plt.plot(
                trajs[i,:,0], trajs[i,:,1], color=color_list[i]
            )

    os.makedirs(save_dir, exist_ok=True)
    vis.save(os.path.join(save_dir, f'{sample_idx}.png'))
    vis.close()


# ===============================
# Main
# ===============================

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    importlib.import_module('utils.nusc_utils')

    dataset = build_dataset(cfg.data.visualization)
    infos = dataset.data_infos
    token2idx = {i['token']: idx for idx, i in enumerate(infos)}

    results = json.load(open(args.result))['results']

    for token in tqdm(results):
        idx = token2idx[token]
        info = infos[idx]

        scene_root = os.path.join(args.show_dir, info['scene_token'])
        visualize_pred(
            idx, token, dataset, info,
            results[token], results, score_thresh,
            os.path.join(scene_root, 'pred')
        )
        visualize_gt(
            idx, dataset, info,
            infos,
            os.path.join(scene_root, 'gt')
        )


if __name__ == '__main__':
    main()
