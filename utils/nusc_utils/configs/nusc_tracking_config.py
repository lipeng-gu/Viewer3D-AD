dataset_type = 'NuScenesTrackingDataset'
data_root = './data/nuscenes/v1.0-trainval/'
class_names = [
    'car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle',
    'pedestrian', 'construction_vehicle', 'traffic_cone', 'barrier'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395], to_rgb=False
)
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

file_client_args = dict(backend='disk')
ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
    }

# Pay attention to how we change the data augmentation
train_pipeline = [
    dict(
        type='LoadPointsFromFile', # visualization purpose
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='TrackLoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_forecasting=True),
    dict(type='TrackInstanceRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='TrackObjectNameFilter', classes=class_names),
]

train_pipeline_multiframe = [
    dict(type='TrackResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='TrackNormalizeMultiviewImage', **img_norm_cfg),
    dict(type='TrackPadMultiViewImage', size_divisor=32),
    dict(type='FormatBundle3DTrack'),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'instance_inds', 'img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'l2g', 'gt_forecasting_locs', 'gt_forecasting_masks', 'gt_forecasting_types'])
]

data = dict(
    visualization=dict(
        type=dataset_type, pipeline=train_pipeline,
        pipeline_multiframe=train_pipeline_multiframe,
        data_root=data_root, test_mode=False, forecasting=True,
        classes=class_names, modality=input_modality,
        ann_file=data_root + 'tracking_forecasting_infos_val.pkl',
        num_frames_per_sample=1
    )
)