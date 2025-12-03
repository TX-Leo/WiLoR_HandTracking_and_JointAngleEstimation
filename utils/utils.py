import os
import cv2
import torch
import subprocess
import numpy as np
from tqdm import tqdm
from typing import Optional

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    return np.array([w_new, h_new])


def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0):
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)),
                               flags=cv2.INTER_LINEAR,
                               borderMode=border_mode,
                               borderValue=border_value,
                               )
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:, :, 3] = cv2.warpAffine(img[:, :, 3], trans, (int(patch_width), int(patch_height)),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT,
                                            )

    return img_patch, trans


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = np.stack([tx, ty, tz], axis=-1)
    return full_cam


import numpy as np


def perspective_projection(points: np.ndarray,
                           translation: np.ndarray,
                           focal_length: np.ndarray,
                           camera_center: Optional[np.ndarray] = None,
                           rotation: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes the perspective projection of a set of 3D points using NumPy.

    Args:
        points (np.ndarray): Array of shape (B, N, 3) containing the input 3D points.
        translation (np.ndarray): Array of shape (B, 3) containing the 3D camera translation.
        focal_length (np.ndarray): Array of shape (B, 2) containing the focal length in pixels.
        camera_center (np.ndarray): Array of shape (B, 2) containing the camera center in pixels.
        rotation (np.ndarray): Array of shape (B, 3, 3) containing the camera rotation.

    Returns:
        np.ndarray: Array of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]

    if rotation is None:
        rotation = np.eye(3)[np.newaxis, :, :].repeat(batch_size, axis=0)
    if camera_center is None:
        camera_center = np.zeros((batch_size, 2))

    # Populate intrinsic camera matrix K.
    K = np.zeros((batch_size, 3, 3))
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = np.einsum('bij,bkj->bki', rotation, points)
    points += translation[:, np.newaxis, :]

    # Apply perspective distortion
    projected_points = points / points[:, :, -1][:, :, np.newaxis]

    # Apply camera intrinsics
    projected_points = np.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz": R = Rz @ Ry @ Rx
    elif order == "xzy": R = Ry @ Rz @ Rx
    elif order == "yxz": R = Rz @ Rx @ Ry
    elif order == "yzx": R = Rx @ Rz @ Ry
    elif order == "zyx": R = Rx @ Ry @ Rz
    elif order == "zxy": R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))

def make_4x4_pose(R, t):
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (torch.tensor([0, 0, 0, 1], device=R.device)
              .reshape(*(1,) * len(dims), 1, 4).expand(*dims, 1, 4))
    return torch.cat([pose_3x4, bottom], dim=-2)

def rotx(theta):
    return torch.tensor([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=torch.float32)
def roty(theta):
    return torch.tensor([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], dtype=torch.float32)
def rotz(theta):
    return torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.float32)

def add_audio_to_video(video_path_no_audio, audio_path, output_video_path):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio file not found at {audio_path}, skipping merge for {os.path.basename(output_video_path)}.")
        os.rename(video_path_no_audio, output_video_path)
        return

    print(f"Adding audio to {os.path.basename(output_video_path)}...")
    command = [
        'ffmpeg', '-i', video_path_no_audio, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_video_path
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        os.remove(video_path_no_audio)
        print(f"  -> Successfully created video with audio: {os.path.basename(output_video_path)}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  -> ERROR: FFmpeg failed to add audio. The video will be silent.")
        if isinstance(e, FileNotFoundError):
            print("  -> FFmpeg command not found. Please make sure FFmpeg is installed and in your system's PATH.")
        else:
            print(f"  -> FFmpeg stderr: {e.stderr.decode()}")
        os.rename(video_path_no_audio, output_video_path)

def create_video_from_frames(rgbs, output_video_path, fps, audio_path=None):
    temp_video_path = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
    h, w, _ = rgbs[0].shape
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)
    for frame_rgb in tqdm(rgbs):
        out.write(frame_rgb)
    out.release()
    add_audio_to_video(temp_video_path, audio_path, output_video_path)