import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

import tempfile
import os.path as osp
import cv2
import csv

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


no_cam = 0
no_trial ="trial3"
task = "static"

local_runtime = True
video_path = '/home/ngouget/Codes/HybrIK/examples/dance_short.mp4'
output_csv = '/home/ngouget/Codes/Results/dance_short.csv'

det_config = '/home/ngouget/Codes/OpenCapBench/mmpose/projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'
pose_config = '/home/ngouget/Codes/OpenCapBench/mmpose/configs/body_2d_keypoint/rtmpose/synthpose/rtmpose-m_8xb512-700e_available-body8_halpe26aug_256x192.py'
pose_checkpoint = '/home/ngouget/Codes/OpenCapBench/mmpose/work_dirs/rtmpose-m_8xb512-700e_available-body8_halpe26aug_256x192/epoch_2.pth'


device = 'cpu'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))

# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

def process_video(video_path, detector, pose_estimator, visualizer, show_interval=1):
    """Process a video file and visualize predicted keypoints."""
    all_keypoints = []
    all_scores = []
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    output_root = '../Results/output_video'
    mmengine.mkdir_or_exist(output_root)
    output_video_path = osp.join(output_root, 'processed_video_2.mp4')
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict bbox
        scope = detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)
        detect_result = inference_detector(detector, frame_rgb)
        pred_instance = detect_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                       pred_instance.scores > 0.3)]
        bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

        # Predict keypoints
        pose_results = inference_topdown(pose_estimator, frame_rgb, bboxes)
        data_samples = merge_data_samples(pose_results)

        print(sum(data_samples.pred_instances.keypoint_scores[0])/len(data_samples.pred_instances.keypoint_scores[0]))
        score_moy = sum(data_samples.pred_instances.keypoint_scores[0])/len(data_samples.pred_instances.keypoint_scores[0]) 
        keypoints = data_samples.pred_instances.keypoints[0]

        if keypoints is not None:
            all_keypoints.append(keypoints.flatten())  
            all_scores.append(score_moy)

        
        # Show the results
        visualizer.add_datasample(
            'result',
            frame_rgb,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=False,
            show=False,
            wait_time=show_interval,
            out_file=None,
            kpt_thr=0.005)
        
        # Retrieve the visualized image
        vis_result = visualizer.get_image()
        
        # Convert image from RGB to BGR for OpenCV
        vis_result_bgr = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)
        
        # Write the frame to the output video
        out.write(vis_result_bgr)
        
        # Display the frame using OpenCV
        cv2.imwrite(f'/home/ngouget/Codes/Results/output_frames/frame_{frame_idx:04d}.png', vis_result_bgr)
        
        # Press 'q' to exit the loop
        if 0xFF == ord('q'):
            break
    
    print(f"Il y a {len(keypoints)} keypoints")
    cap.release()
    out.release()
    # print(f"Processed video saved at {output_video_path}")
    # Écriture des keypoints dans un fichier CSV
    
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Écrire les keypoints
        for keypoints, score in zip(all_keypoints, all_scores):
            writer.writerow([score] +list(keypoints) )
    
    print(f"Keypoints saved to {output_csv}")

# Call the function with the video file path
process_video(
    video_path,
    detector,
    pose_estimator,
    visualizer,
    show_interval=1
)
