'''
-----------------------
| Blake Troutman 2025 |
-----------------------


This script is a hackathon project to densify a COLMAP SFM reconstruction using MASt3R.

It is designed to be a simple script that can be run on a single machine with a GPU.

It is not designed to be a production-ready script.

Since this script is a hackathon project, manage your expectations accordingly when parsing through the code.


Any bad code you find here is the result of Cursor; all the good code is mine. 

'''

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import pycolmap
from pathlib import Path
import json
import random
import torch
from PIL import Image
import cv2
import time
import argparse

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r 
from dust3r.utils.image import load_images
from dust3r.inference import inference

NO_POINT = 18446744073709551615
DEFAULT_OUTPUT = 'densified.ply'
DEFAULT_PAIRS = 'pairs.json'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reconstruction_path', type=str, required=True, help='Path to the COLMAP reconstruction directory', default='/home/blakedd/data/7ee503023c_402121BD38OPENPIPELINE_HOLES/689a20d452eaff0aa5589bd5/sparse/')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to the directory containing the images', default='/home/blakedd/data/7ee503023c_402121BD38OPENPIPELINE_HOLES/7ee503023c_402121BD38OPENPIPELINE_HOLES/images/')
    parser.add_argument('--output_path', type=str, required=False, help='Path and name of the output PLY file', default=DEFAULT_OUTPUT)
    parser.add_argument('--pairs_path', type=str, required=False, help='Path and name of the output pairs JSON file', default=DEFAULT_PAIRS)
    parser.add_argument('--use_existing_pairs', action='store_true', help='Use existing pairs from JSON file if it exists')
    parser.add_argument('--batch_size', type=int, required=False, help='Batch size for inference. 1 will always work, but will be slower. 24GB GPU can handle batch size of 30', default=1)
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU inference instead of CUDA')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--sampling_factor', type=int, required=False, help='Sampling factor for point triangulation. Lower = denser. User powers of 2. Default = 8', default=8)
    parser.add_argument('--min_feature_coverage', type=float, required=False, help='Minimum proportion of image that must be covered by shared points to be considered a good match. Default = 0.6', default=0.6)
    args = parser.parse_args()

    reconstruction_path = args.reconstruction_path
    img_dir = args.img_dir
    output_path = args.output_path
    pairs_path = args.pairs_path
    use_existing_pairs = args.use_existing_pairs
    batch_size = args.batch_size
    force_cpu = args.force_cpu
    verbose = args.verbose
    sampling_factor = args.sampling_factor
    min_feature_coverage = args.min_feature_coverage

    if use_existing_pairs and not os.path.exists(pairs_path):
        print(f"Pairs file {pairs_path} does not exist. Falling back to generating new pairs.")
        use_existing_pairs = False

    # output path may include a filename. create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # if output path is just a directory, add the default filename
    if not output_path.endswith('.ply'):
        output_path = os.path.join(output_path, DEFAULT_OUTPUT)

    torch.cuda.empty_cache()
    start_time = time.time()

    print(f"Loading reconstruction from {reconstruction_path}...")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    
    if not use_existing_pairs:
        pairs = get_best_pairs(reconstruction, min_feature_coverage=min_feature_coverage)
        with open(pairs_path, 'w') as f:
            print(f"Saving pairs to {pairs_path}...")
            json.dump(pairs, f, indent=4)
    else:
        pairs = json.load(open(pairs_path))
        print(f"Loaded pairs from {pairs_path}...")

    densified_pairs = densify_pairs_mast3r_batch(reconstruction, img_dir, pairs, batch_size=batch_size, sampling_factor=sampling_factor, force_cpu=force_cpu, verbose=verbose)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # display dense clouds in open3d
    all_points = None
    all_colors = None
    for img1, densified in densified_pairs.items():
        if all_points is None:
            all_points = densified['points']
            all_colors = densified['colors']
        else:
            all_points = np.concatenate([all_points, densified['points']], axis=0)
            all_colors = np.concatenate([all_colors, densified['colors']], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # save all points to a ply file
    o3d.io.write_point_cloud(output_path, pcd)

    # display the point cloud
    o3d.visualization.draw_geometries([pcd])


def densify_pairs_mast3r_batch(reconstruction, img_dir, pairs, batch_size=20, sampling_factor=8, force_cpu=False, verbose=False) -> dict[int, np.ndarray]:

    # fixed params for MASt3R
    model_w = 512
    model_h = 384
    size = 512
    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the model
    model = AsymmetricMASt3R.from_pretrained(model_path, verbose=verbose).to(device)

    # initialize empty dict of image_id to np.ndarray of point3D_ids
    densified_pairs = {}

    def densify_batch(pair_batch):
        image_batch = []
        image_batch_sizes = []
        img_id_to_idx = {}
        image_paths = []

        # get all image indices that appear in pair_batch
        image_indices = set()
        for img1, img2 in pair_batch.items():
            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue
            image_indices.add(int(img1))
            image_indices.add(int(img2))

        # Skip this batch if no valid images found
        if not image_indices:
            print("Skipping batch - no valid images found in reconstruction...")
            return

        # get the image paths for the image indices
        for image_idx in image_indices:
            img_name = reconstruction.images[image_idx].name
            img_path = os.path.join(img_dir, img_name)
            image_paths.append(img_path)
            img_id_to_idx[image_idx] = len(image_paths) - 1

        # load all images in the batch
        print("loading images...")
        images = load_images(image_paths, size=size)

        # pair up the images
        for img1, img2 in pair_batch.items():
            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue
            img1_idx = img_id_to_idx[int(img1)]
            img2_idx = img_id_to_idx[int(img2)]
            image_batch.append(tuple([images[img1_idx], images[img2_idx]]))
            cam_1 = reconstruction.images[int(img1)].camera
            cam_2 = reconstruction.images[int(img2)].camera
            image_batch_sizes.append([tuple([cam_1.width, cam_1.height]), tuple([cam_2.width, cam_2.height])])

        # get predictions for each image pair
        output = inference(image_batch, model=model, device=device, batch_size=batch_size, verbose=verbose)

        # triangulate each image pair
        print("triangulating...")
        for i in range(len(image_batch)):
            img1, img2 = list(pair_batch.items())[i]

            if int(img1) not in reconstruction.images or int(img2) not in reconstruction.images:
                continue

            view1, pred1 = output['view1'], output['pred1']
            view2, pred2 = output['view2'], output['pred2']

            desc1, desc2 = pred1['desc'][i].squeeze(0).detach(), pred2['desc'][i].squeeze(0).detach()

            matches_img1, matches_img2 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=sampling_factor, device=device, dist='dot', block_size=2**13)

            # ignore small border around the edge
            H1, W1 = view1['true_shape'][i]
            valid_matches_img1 = (matches_img1[:, 0] >= 3) & (matches_img1[:, 1] >= 3) & (matches_img1[:, 0] < int(W1) - 3) & (matches_img1[:, 1] < int(H1) - 3)

            H2, W2 = view2['true_shape'][i]
            valid_matches_img2 = (matches_img2[:, 0] >= 3) & (matches_img2[:, 1] >= 3) & (matches_img2[:, 0] < int(W2) - 3) & (matches_img2[:, 1] < int(H2) - 3)

            valid_matches = valid_matches_img1 & valid_matches_img2

            matches_img1 = matches_img1[valid_matches]
            matches_img2 = matches_img2[valid_matches]

            if matches_img1.shape[0] < 1 or matches_img2.shape[0] < 1:
                continue

            # rescale matches to original image size
            w1, h1 = image_batch_sizes[i][0]
            w1_scale = w1 / model_w
            h1_scale = h1 / model_h
            w2, h2 = image_batch_sizes[i][1]
            w2_scale = w2 / model_w
            h2_scale = h2 / model_h

            matches_img1[:, 0] = matches_img1[:, 0] * w1_scale
            matches_img1[:, 1] = matches_img1[:, 1] * h1_scale
            matches_img2[:, 0] = matches_img2[:, 0] * w2_scale
            matches_img2[:, 1] = matches_img2[:, 1] * h2_scale

            # get colors from image 1
            img1_path = os.path.join(img_dir, reconstruction.images[int(img1)].name)
            img1_color = np.zeros((len(matches_img1), 3))
            image = Image.open(img1_path)
            img_width, img_height = image.size  # Get actual image dimensions
            
            for j in range(len(matches_img1)):
                x, y = matches_img1[j]
                # Convert to integers and clamp to image bounds
                x = int(np.clip(x, 0, img_width - 1))
                y = int(np.clip(y, 0, img_height - 1))
                img1_color[j] = np.array(image.getpixel((x, y)), dtype=np.float32) / 255.0

            # get triangulated points
            K1 = reconstruction.images[int(img1)].camera.calibration_matrix()
            K2 = reconstruction.images[int(img2)].camera.calibration_matrix()
            P1 = K1 @ reconstruction.images[int(img1)].cam_from_world().matrix()
            P2 = K2 @ reconstruction.images[int(img2)].cam_from_world().matrix()

            m1_undistort = matches_img1.astype(np.float64).T
            m2_undistort = matches_img2.astype(np.float64).T

            # get distortion parameters
            dist_keys_1 = reconstruction.images[int(img1)].camera.params_info.split(', ')
            dist_keys_2 = reconstruction.images[int(img2)].camera.params_info.split(', ')
            dist1 = {}
            dist2 = {}
            for j in range(len(dist_keys_1)):
                dist1[dist_keys_1[j]] = reconstruction.images[int(img1)].camera.params[j]
            for j in range(len(dist_keys_2)):
                dist2[dist_keys_2[j]] = reconstruction.images[int(img2)].camera.params[j]

            dist_coeffs_1 = np.array([dist1.get('k1', 0), dist1.get('k2', 0), dist1.get('p1', 0), dist1.get('p2', 0), dist1.get('k3', 0), dist1.get('k4', 0), dist1.get('k5', 0), dist1.get('k6', 0)])
            dist_coeffs_2 = np.array([dist2.get('k1', 0), dist2.get('k2', 0), dist2.get('p1', 0), dist2.get('p2', 0), dist2.get('k3', 0), dist2.get('k4', 0), dist2.get('k5', 0), dist2.get('k6', 0)])

            # undistort the matches
            m1_undistort = cv2.undistortPoints(m1_undistort, K1, dist_coeffs_1, P=K1)
            m2_undistort = cv2.undistortPoints(m2_undistort, K2, dist_coeffs_2, P=K2)

            triangulated_points = cv2.triangulatePoints(P1, P2, m1_undistort, m2_undistort)
            triangulated_points = triangulated_points / triangulated_points[3, :]
            triangulated_points = triangulated_points[:3, :]
            triangulated_points = triangulated_points.transpose()

            # filter out points that are too far from the cameras
            img1_cam = reconstruction.images[int(img1)].cam_from_world()
            img1_R = img1_cam.rotation.matrix()
            img1_t = img1_cam.translation

            img2_cam = reconstruction.images[int(img2)].cam_from_world()
            img2_R = img2_cam.rotation.matrix()
            img2_t = img2_cam.translation

            # image center of img1
            img1_C = -img1_R.T @ img1_t

            # image center of img2
            img2_C = -img2_R.T @ img2_t

            baseline = np.linalg.norm(img1_C - img2_C)

            # filter out points that are too far from the cameras (parallax stand-in)
            dist_to_cameras = np.linalg.norm(triangulated_points - img1_C, axis=1)
            triangulated_points = triangulated_points[dist_to_cameras < 20 * baseline, :]
            img1_color = img1_color[dist_to_cameras < 20 * baseline]

            # filter out points that are behind the camera
            behind_camera = (img1_R @ triangulated_points.T - img1_t.reshape(-1, 1))[2] < 0
            triangulated_points = triangulated_points[~behind_camera, :]
            img1_color = img1_color[~behind_camera]

            # TODO: filter out points that violate epipolar constraint

            # # [OPTIONAL] add the image point to the triangulated points
            # triangulated_points = np.concatenate([triangulated_points, img1_C.reshape(-1, 3)], axis=0)
            # img1_color = np.concatenate([img1_color, np.array([1, 0, 0]).reshape(1, 3)], axis=0)

            densified_pairs[img1] = {
                'points': triangulated_points,
                'colors': img1_color
            }

    num_batches = len(pairs) // batch_size
    batches = []
    for i in range(num_batches):
        batch = {k: v for k, v in pairs.items() if k in list(pairs.keys())[i*batch_size:(i+1)*batch_size]}
        batches.append(batch)

    for batch in batches:
        densify_batch(batch)
    
    return densified_pairs


def get_best_pairs(reconstruction, min_points=100, parallax_sample_size=100, min_feature_coverage=0.6):

    class MatchCandidate:
        def __init__(self, image_id, shared_points):
            self.image_id = image_id
            self.shared_points = shared_points
            self.avg_parallax_angle = 0
            self.x_coverage = 0
            self.y_coverage = 0

    # initialize empty pair map (image ids)
    pairs = {}
    for image in reconstruction.images.values():
        pairs[image.image_id] = -1
    
    # generate dict of image_id to set of point3D_ids
    image_point3D_ids = {}
    image_point3D_xy = {}
    for image in reconstruction.images.values():
        image_point3D_xy[image.image_id] = {}
        for point2D in image.points2D:
            if point2D.has_point3D():
                image_point3D_xy[image.image_id][point2D.point3D_id] = point2D.xy
                if image_point3D_ids.get(image.image_id, None) is None:
                    image_point3D_ids[image.image_id] = set()
                image_point3D_ids[image.image_id].add(point2D.point3D_id)

    # iterate over all images
    for i, image in enumerate(reconstruction.images.values()):
        print(f"Getting best corresponding image for {image.name}... {i}/{len(reconstruction.images) - 1}")
        # identify other images that share at least 100 points
        other_images = [other_image for other_image in reconstruction.images.values() if other_image.image_id != image.image_id]
        match_candidates = []
        for other_image in other_images:
            # get the points that the two images share
            shared_points = image_point3D_ids[image.image_id] & image_point3D_ids[other_image.image_id]
            match_candidates.append(MatchCandidate(other_image.image_id, shared_points))
        
        # sort by number of shared points
        match_candidates.sort(key=lambda x: len(x.shared_points), reverse=True)
        
        # if there are no frames with a good number of shared points, skip this one
        if len(match_candidates) == 0 or len(match_candidates[0].shared_points) < min_points:
            continue
        
        # filter out match candidates that don't have enough matches
        match_candidates = [match_candidate for match_candidate in match_candidates if len(match_candidate.shared_points) >= min_points]

        # for each match candidate, compute the feature coverage of the shared points
        for match_candidate in match_candidates:
            shared_points = match_candidate.shared_points
            xs = [image_point3D_xy[image.image_id][point_id][0] for point_id in shared_points]
            ys = [image_point3D_xy[image.image_id][point_id][1] for point_id in shared_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x_range = x_max - x_min
            y_range = y_max - y_min
            match_candidate.x_coverage = x_range / image.camera.width
            match_candidate.y_coverage = y_range / image.camera.height

        # order match candidates by average xy coverage
        match_candidates.sort(key=lambda x: (x.x_coverage + x.y_coverage) / 2, reverse=True)
        
        # # filter out match candidates that don't have enough feature coverage in the base image
        # match_candidates = [match_candidate for match_candidate in match_candidates if match_candidate.x_coverage > min_feature_coverage and match_candidate.y_coverage > min_feature_coverage]

        # for each match candidate, record the average parallax angle between the two images and each shared point
        for match_candidate in match_candidates:
            other_image = reconstruction.images[match_candidate.image_id]
            shared_points = match_candidate.shared_points
            parallax_angles = []
            for point_id in random.sample(list(shared_points), parallax_sample_size):
                point = reconstruction.points3D[point_id]
                img1_cam = reconstruction.images[image.image_id].cam_from_world()
                img1_R = img1_cam.rotation.matrix()
                img1_t = img1_cam.translation
                img2_cam = reconstruction.images[other_image.image_id].cam_from_world()
                img2_R = img2_cam.rotation.matrix()
                img2_t = img2_cam.translation
                img1_C = -img1_R.T @ img1_t
                img2_C = -img2_R.T @ img2_t
                u = img1_C - point.xyz
                v = img2_C - point.xyz
                parallax_angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                parallax_angles.append(parallax_angle)
            
            avg_parallax_angle = np.mean(parallax_angles)
            match_candidate.avg_parallax_angle = avg_parallax_angle

        if len(match_candidates) == 0:
            continue
        
        # make sure the high overlap match has good parallax
        winning_index = -1
        for i in range(len(match_candidates)):
            if match_candidates[i].avg_parallax_angle > 0.1:
                winning_index = i
                break
        if winning_index == -1:
            winning_index = 0
        
        pairs[image.image_id] = match_candidates[winning_index].image_id

    return pairs

if __name__ == "__main__":
    main()