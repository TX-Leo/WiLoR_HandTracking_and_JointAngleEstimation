# -*- coding: utf-8 -*-
"""
wilor_v2.py

本脚本实现了对视频进行手部姿态估计的完整流程，使用了 WiLoR 模型。

【版本更新 v3】
1. 修复 Pose 可视化：改用基于3D关键点的几何法构建坐标轴，解决轴向错误问题。
2. 优化 Dashboard：左右两侧固定面板，分别显示左手/右手状态 (Missing/Detected)。
3. 增加 Abduction 可视化：仪表盘现在同时显示弯曲(Flexion)和外展(Abduction)。

主要功能包括：
1. 使用 WiLoR 模型对视频的每一帧进行3D手部姿态和形状的估计。
2. 将估计出的3D手部模型渲染回2D图像。
3. 提取并保存 Joint Angles (emg2pose定义) 和 Confidence。
4. 生成包含丰富信息的数据仪表盘视频。
"""

import os

# ==============================================================================
# 【环境配置】解决 Pyrender 在无头服务器上的渲染问题
# 必须在 import pyrender 之前设置
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# ==============================================================================

import cv2
import json
import torch
import trimesh
import pyrender
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

from utils import utils

from pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# MANO模型关键点连接顺序 (基于用户提供的映射逻辑)
MANO_CONNECTIONS = [
    # 拇指 Thumb (索引 1-4)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 无名指 Ring (索引 5-8) - 注：此处遵循用户提供的索引定义
    (0, 5), (5, 6), (6, 7), (7, 8),
    # 中指 Middle (索引 9-12)
    (0, 9), (9, 10), (10, 11), (11, 12),
    # 食指 Index (索引 13-16)
    (0, 13), (13, 14), (14, 15), (15, 16),
    # 小指 Pinky (索引 17-20)
    (0, 17), (17, 18), (18, 19), (19, 20),
]

class Renderer:
    def __init__(self, faces: np.array):
        # 补充闭合面片
        faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), rot_axis=[1, 0, 0], rot_angle=0, is_right=1):
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy() if is_right else self.faces_left.copy(), vertex_colors=vertex_colors)
        rot = trimesh.transformations.rotation_matrix(np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(self, vertices: np.array, cam_t=None, rot=None, rot_axis=[1, 0, 0], rot_angle=0, camera_z=3, mesh_base_color=(1.0, 1.0, 0.9), scene_bg_color=(0, 0, 0), render_res=[256, 256], focal_length=None, is_right=None):
        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0], viewport_height=render_res[1], point_size=1.0)
        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])
        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=render_res[0] / 2., cy=render_res[1] / 2., zfar=1e12)
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)
        for node in self.create_raymond_lights(): scene.add_node(node)
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()
        return color.astype(np.float32) / 255.0

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        for i, pose in enumerate(self.get_light_poses() + [np.eye(4)]):
            matrix = scene.get_pose(cam_node) @ pose
            node = pyrender.Node(name=f"light-{i:02d}", light=pyrender.DirectionalLight(color=color, intensity=intensity), matrix=matrix)
            if not scene.has_node(node): scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        for i, pose in enumerate(self.get_light_poses(dist=0.5) + [np.eye(4)]):
            matrix = scene.get_pose(cam_node) @ pose
            node = pyrender.Node(name=f"plight-{i:02d}", light=pyrender.PointLight(color=color, intensity=intensity), matrix=matrix)
            if not scene.has_node(node): scene.add_node(node)
    
    def create_raymond_lights(self):
        thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
        nodes = []
        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            yp = np.sin(theta) * np.sin(phi)
            zp = np.cos(theta)
            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)
            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix
            ))
        return nodes

    def get_light_poses(self, n_lights=5, elevation=np.pi / 3, dist=12):
        thetas = elevation * np.ones(n_lights)
        phis = 2 * np.pi * np.arange(n_lights) / n_lights
        poses = []
        trans = utils.make_translation(torch.tensor([0, 0, dist]))
        for phi, theta in zip(phis, thetas):
            rot = utils.make_rotation(rx=-theta, ry=phi, order="xyz")
            poses.append((rot @ trans).numpy())
        return poses
    
@dataclass
class WilorHand:
    """存储单只手的检测结果。"""
    is_right: bool = False
    hand_keypoints_2d: np.ndarray = None
    hand_keypoints_3d: np.ndarray = None
    vertices: np.ndarray = None
    global_orient: np.ndarray = None
    hand_pose: np.ndarray = None
    cam_t: np.ndarray = None
    bbox: np.ndarray = None
    confidence: float = 1.0
    joint_angles: Dict[str, float] = field(default_factory=dict)

@dataclass
class WilorStruct:
    """用于存储单帧图像所有检测信息的抽象数据结构。"""
    idx: int
    rgb: np.ndarray
    focal_length: float = None
    hands: List[WilorHand] = field(default_factory=list)

class WilorDataset:
    def __init__(self, video_path: str, save_path: str):
        self.video_path = video_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频加载成功: {self.width}x{self.height} @ {self.fps} FPS, 共 {self.num_frames} 帧。")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f"正在加载 WiLoR 模型到 {self.device}...")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=self.dtype)
        print("WiLoR 模型加载完毕。")

        self.renderer = Renderer(self.pipe.wilor_model.mano.faces)
        self.RIGHT_HAND_COLOR = (0.66, 0.27, 0.25)
        self.LEFT_HAND_COLOR = (0.2, 0.2, 0.6)

        self.all_wilor_structs: List[WilorStruct] = []

    def __len__(self) -> int:
        return self.num_frames

    def _calculate_emg2pose_angles(self, keypoints_3d: np.ndarray, is_right: bool) -> Dict[str, float]:
        """计算 emg2pose 论文定义的 20 个关节角度。"""
        angles = {}
        def get_angle(v1, v2):
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
            dot = np.dot(v1_norm, v2_norm)
            dot = np.clip(dot, -1.0, 1.0)
            return np.degrees(np.arccos(dot))

        def get_abduction(bone_vec, ref_vec, plane_normal):
            def project(v): return v - np.dot(v, plane_normal) * plane_normal
            v_proj = project(bone_vec)
            ref_proj = project(ref_vec)
            return get_angle(v_proj, ref_proj)

        kpts = keypoints_3d
        v_wrist_middle = kpts[9] - kpts[0] 
        v_wrist_index = kpts[13] - kpts[0]
        # 注意：法线方向可能随左右手变化，但在计算无符号角度时影响较小，主要影响外展正负
        palm_normal = np.cross(v_wrist_index, v_wrist_middle)
        palm_normal /= (np.linalg.norm(palm_normal) + 1e-6)
        
        fingers_map = {
            'Index': [13, 14, 15, 16],
            'Middle': [9, 10, 11, 12],
            'Ring': [5, 6, 7, 8],
            'Pinky': [17, 18, 19, 20]
        }
        v_middle_proximal = kpts[10] - kpts[9]

        for name, idxs in fingers_map.items():
            mcp, pip, dip, tip = idxs
            v_metacarpal = kpts[mcp] - kpts[0]
            v_proximal = kpts[pip] - kpts[mcp]
            v_intermediate = kpts[dip] - kpts[pip]
            v_distal = kpts[tip] - kpts[dip]
            
            angles[f'{name}_MCP_Flexion'] = get_angle(v_metacarpal, v_proximal)
            if name == 'Middle':
                 angles[f'{name}_MCP_Abduction'] = 0.0
            else:
                 angles[f'{name}_MCP_Abduction'] = get_abduction(v_proximal, v_middle_proximal, palm_normal)

            angles[f'{name}_PIP_Flexion'] = get_angle(v_proximal, v_intermediate)
            angles[f'{name}_DIP_Flexion'] = get_angle(v_intermediate, v_distal)

        v_ref_thumb = kpts[1] - kpts[0]
        v_thumb_metacarpal = kpts[2] - kpts[1]
        v_thumb_proximal = kpts[3] - kpts[2]
        v_thumb_distal = kpts[4] - kpts[3]
        
        angles['Thumb_CMC_Flexion'] = get_angle(v_ref_thumb, v_thumb_metacarpal)
        angles['Thumb_CMC_Abduction'] = get_abduction(v_thumb_metacarpal, v_middle_proximal, palm_normal)
        angles['Thumb_MCP_Flexion'] = get_angle(v_thumb_metacarpal, v_thumb_proximal)
        angles['Thumb_IP_Flexion'] = get_angle(v_thumb_proximal, v_thumb_distal)

        return angles

    def __getitem__(self, idx: int) -> WilorStruct:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = self.cap.read()
        if not ret:
            raise IndexError(f"无法读取第 {idx} 帧。")

        outputs = self.pipe.predict(frame_bgr)
        wilor_hands = []
        focal_length = None

        for out in outputs:
            wilor_preds = out['wilor_preds']
            is_right = out['is_right']
            focal_length = wilor_preds['scaled_focal_length']
            
            bbox = out['hand_bbox']
            confidence = 1.0
            # 兼容不同版本的返回结构
            if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 5:
                confidence = float(bbox[4])
                bbox = bbox[:4]
            elif 'score' in out:
                confidence = float(out['score'])
            elif 'conf' in out:
                confidence = float(out['conf'])
            
            kpts_3d = wilor_preds["pred_keypoints_3d"][0]
            angles = self._calculate_emg2pose_angles(kpts_3d, is_right)

            hand_data = WilorHand(
                is_right=is_right,
                hand_keypoints_2d=wilor_preds["pred_keypoints_2d"][0],
                hand_keypoints_3d=kpts_3d,
                vertices=wilor_preds['pred_vertices'][0],
                global_orient=wilor_preds['global_orient'][0],
                hand_pose=wilor_preds['hand_pose'][0],
                cam_t=wilor_preds['pred_cam_t_full'][0],
                bbox=np.array(bbox),
                confidence=confidence,
                joint_angles=angles
            )
            wilor_hands.append(hand_data)

        return WilorStruct(
            idx=idx,
            rgb=frame_bgr,
            focal_length=focal_length,
            hands=wilor_hands
        )
        
    def process_and_save_all(self):
        print("开始处理所有视频帧...")
        for i in tqdm(range(len(self)), desc="处理视频帧"):
            wilor_struct = self[i]
            self.all_wilor_structs.append(wilor_struct)
            self.save_struct_to_json(wilor_struct)

        print("所有帧处理完毕，开始生成可视化视频...")
        self.create_visualization_videos()
        print("所有任务完成！")

    def save_struct_to_json(self, wilor_struct: WilorStruct):
        frame_dir = os.path.join(self.save_path, "all_data", f"{wilor_struct.idx:05d}")
        os.makedirs(frame_dir, exist_ok=True)

        def to_list_safe(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            return item

        data_to_save = {
            "frame_index": int(wilor_struct.idx),
            "focal_length": float(wilor_struct.focal_length) if wilor_struct.focal_length is not None else None,
            "hands": []
        }
        for hand in wilor_struct.hands:
            safe_joint_angles = {k: float(v) for k, v in hand.joint_angles.items()}
            hand_dict = {
                "is_right": bool(hand.is_right),
                "confidence": float(hand.confidence),
                "bbox": to_list_safe(hand.bbox),
                "joint_angles_emg2pose": safe_joint_angles,
                "hand_keypoints_2d": to_list_safe(hand.hand_keypoints_2d),
                "hand_keypoints_3d": to_list_safe(hand.hand_keypoints_3d),
                "global_orient(axis_angle)": to_list_safe(hand.global_orient),
                "hand_pose(axis_angle)": to_list_safe(hand.hand_pose),
                "camera_translation": to_list_safe(hand.cam_t),
            }
            data_to_save["hands"].append(hand_dict)
            
        json_path = os.path.join(frame_dir, "data.json")
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
    def create_visualization_videos(self):
        if not self.all_wilor_structs:
            print("没有已处理的数据，无法创建视频。")
            return

        rendered_hand_frames, rendered_hand_mask_only_frames, palm_and_wrist_pose_frames, hand_keypoints_frames, all_info_frames = [], [], [], [], []

        for s in tqdm(self.all_wilor_structs, desc="生成可视化帧"):
            image_rgb_norm = cv2.cvtColor(s.rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rendered_hand_mask_only_canvas = np.zeros_like(image_rgb_norm)
            rendered_hand_canvas = image_rgb_norm.copy()
            
            for hand in s.hands:
                mesh_color = self.RIGHT_HAND_COLOR if hand.is_right else self.LEFT_HAND_COLOR
                rendered_rgba = self.renderer.render_rgba(
                    vertices=hand.vertices, 
                    cam_t=hand.cam_t,
                    render_res=[self.width, self.height], 
                    is_right=hand.is_right,
                    mesh_base_color=mesh_color,
                    scene_bg_color=(0, 0, 0),
                    focal_length=s.focal_length
                )
                alpha = rendered_rgba[:, :, 3:]
                rendered_hand_mask_only_canvas = rendered_hand_mask_only_canvas * (1 - alpha) + rendered_rgba[:, :, :3] * alpha
                rendered_hand_canvas = rendered_hand_canvas * (1 - alpha) + rendered_rgba[:, :, :3] * alpha

            # 基础渲染帧
            base_rendered = cv2.cvtColor((rendered_hand_canvas * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            rendered_hand_frames.append(base_rendered)
            rendered_hand_mask_only_frames.append(cv2.cvtColor((rendered_hand_mask_only_canvas * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            
            # Pose 帧
            pose_frame = self.draw_palm_and_wrist_pose(s.rgb.copy(), s.hands, s.focal_length, self.width, self.height)
            palm_and_wrist_pose_frames.append(pose_frame)
            
            # Keypoints 帧
            kpt_frame = self._draw_hand_keypoints(s.rgb.copy(), s.hands)
            hand_keypoints_frames.append(kpt_frame)
            
            # All Info 帧 (包含 Pose, Kpts, Mesh, Dashboard)
            img_step1 = self.draw_palm_and_wrist_pose(base_rendered.copy(), s.hands, s.focal_length, self.width, self.height)
            img_step2 = self._draw_hand_keypoints(img_step1, s.hands)
            # 【新功能】绘制固定布局的左右手 Dashboard
            img_final = self._draw_angle_dashboard_fixed_layout(img_step2, s.hands)
            
            all_info_frames.append(img_final)

        utils.create_video_from_frames(rendered_hand_frames, os.path.join(self.save_path, f"viz_rendered_hand.mp4"), self.fps)
        utils.create_video_from_frames(rendered_hand_mask_only_frames, os.path.join(self.save_path, f"viz_rendered_hand_mask_only.mp4"), self.fps)
        utils.create_video_from_frames(palm_and_wrist_pose_frames, os.path.join(self.save_path, f"viz_palm_and_wrist_pose.mp4"), self.fps)
        utils.create_video_from_frames(hand_keypoints_frames, os.path.join(self.save_path, f"viz_hand_keypoints.mp4"), self.fps)
        utils.create_video_from_frames(all_info_frames, os.path.join(self.save_path, f"viz_all_info.mp4"), self.fps)

    # --- 【重构】固定布局的 Dashboard ---
    def _draw_angle_dashboard_fixed_layout(self, image: np.ndarray, hands: List[WilorHand]) -> np.ndarray:
        """
        绘制左右两个固定的面板，分别显示左右手的 Flexion 和 Abduction 状态。
        如果没有检测到手，面板依然存在但显示"Not Detected"。
        """
        h, w = image.shape[:2]
        overlay = image.copy()
        
        panel_w = 280
        panel_h = 240
        margin = 15
        
        # 定义左右面板的位置
        # Left Panel (for Left Hand)
        left_panel_rect = (margin, h - panel_h - margin, margin + panel_w, h - margin)
        # Right Panel (for Right Hand)
        right_panel_rect = (w - panel_w - margin, h - panel_h - margin, w - margin, h - margin)
        
        # 查找当前帧中的左手和右手数据
        left_hand_data = next((h for h in hands if not h.is_right), None)
        right_hand_data = next((h for h in hands if h.is_right), None)
        
        # 绘制面板背景
        cv2.rectangle(overlay, (left_panel_rect[0], left_panel_rect[1]), (left_panel_rect[2], left_panel_rect[3]), (0, 0, 0), -1)
        cv2.rectangle(overlay, (right_panel_rect[0], right_panel_rect[1]), (right_panel_rect[2], right_panel_rect[3]), (0, 0, 0), -1)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # 辅助绘制函数
        def draw_panel_content(start_x, start_y, hand: Optional[WilorHand], title_prefix):
            # 标题
            cv2.putText(image, f"{title_prefix} Hand", (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if hand is None:
                # 没检测到手，显示灰色提示
                cv2.putText(image, "Not Detected", (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                return

            # 显示置信度
            cv2.putText(image, f"Conf: {hand.confidence:.2f}", (start_x + 140, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            bar_start_y = start_y + 50
            row_h = 35 # 每行高度增加，因为要画两条条
            
            for i, finger in enumerate(fingers):
                y = bar_start_y + i * row_h
                
                # 1. 提取 Flexion (平均值)
                flex_keys = [k for k in hand.joint_angles.keys() if finger in k and "Flexion" in k]
                avg_flex = np.mean([hand.joint_angles[k] for k in flex_keys]) if flex_keys else 0
                
                # 2. 提取 Abduction (平均值, 对于中指可能是0)
                abd_keys = [k for k in hand.joint_angles.keys() if finger in k and "Abduction" in k]
                avg_abd = np.mean([hand.joint_angles[k] for k in abd_keys]) if abd_keys else 0
                
                # 绘制文字
                cv2.putText(image, f"{finger[:3]}", (start_x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 绘制 Flexion 条 (蓝色, 上半部分)
                # 范围假设 0-100度
                flex_ratio = min(max(avg_flex / 100.0, 0), 1.0)
                bar_len_f = int(flex_ratio * (panel_w - 70))
                cv2.rectangle(image, (start_x + 55, y + 5), (start_x + 55 + bar_len_f, y + 15), (255, 150, 50), -1) # Blue-ish (BGR)
                
                # 绘制 Abduction 条 (黄色, 下半部分, 较细)
                # 范围假设 0-40度
                abd_ratio = min(max(avg_abd / 40.0, 0), 1.0)
                bar_len_a = int(abd_ratio * (panel_w - 70))
                cv2.rectangle(image, (start_x + 55, y + 18), (start_x + 55 + bar_len_a, y + 24), (50, 200, 255), -1) # Yellow (BGR)
                
                # 数字显示 (只显示Flexion方便阅读，或者全部显示)
                # F: Flexion, A: Abduction
                # cv2.putText(image, f"{int(avg_flex)}", (start_x + panel_w - 30, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 图例
            legend_y = start_y + panel_h - 15
            cv2.rectangle(image, (start_x + 10, legend_y), (start_x + 20, legend_y+10), (255, 150, 50), -1)
            cv2.putText(image, "Flex", (start_x + 25, legend_y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.rectangle(image, (start_x + 70, legend_y), (start_x + 80, legend_y+10), (50, 200, 255), -1)
            cv2.putText(image, "Abd", (start_x + 85, legend_y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 绘制左边面板
        draw_panel_content(left_panel_rect[0], left_panel_rect[1], left_hand_data, "Left")
        # 绘制右边面板
        draw_panel_content(right_panel_rect[0], right_panel_rect[1], right_hand_data, "Right")

        return image

    # --- 【重构】基于几何关键点的 Pose 可视化 ---
    def draw_palm_and_wrist_pose(self, image: np.ndarray, hands: List[WilorHand], focal_length, width, height):
        """
        使用 3D 关键点直接构建手掌局部坐标系，避免因 Global Orient 定义差异导致的坐标轴歪斜。
        X轴: 红色 (Thumb/Index direction)
        Y轴: 绿色 (Middle finger direction)
        Z轴: 蓝色 (Palm Normal, out of palm)
        """
        if focal_length is None or not hands:
            return image
        
        camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # 这里的 R_correction 用来修正 WiLoR 相机坐标系(Y-up) 到 OpenCV (Y-down) 的差异
        # 仅用于平移向量的投影修正，旋转矩阵我们将自己构建，所以不需要在这个矩阵里旋转
        R_correction = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

        axis_len = 0.08 # 8cm 长

        for hand in hands:
            kpts = hand.hand_keypoints_3d # (21, 3)
            
            # --- 1. 使用几何法构建局部坐标系 ---
            # 原点：手腕 (Index 0)
            origin = kpts[0]
            
            # Y轴向量：手腕 -> 中指指根 (Index 9) [纵向]
            vec_y = kpts[9] - kpts[0]
            vec_y = vec_y / (np.linalg.norm(vec_y) + 1e-6)
            
            # 辅助向量：手腕 -> 食指指根 (Index 13)
            vec_index = kpts[13] - kpts[0]
            
            # Z轴向量 (法线)：Cross(Index, Middle) -> 垂直于手掌平面
            # 注意：左右手叉乘方向可能相反，这里统一处理，如果发现Z轴反了可以加判断
            if hand.is_right:
                vec_z = np.cross(vec_index, vec_y) # 右手定则
            else:
                vec_z = np.cross(vec_y, vec_index) # 左手反过来
            vec_z = vec_z / (np.linalg.norm(vec_z) + 1e-6)
            
            # X轴向量：Cross(Y, Z) -> 确保正交
            vec_x = np.cross(vec_y, vec_z)
            vec_x = vec_x / (np.linalg.norm(vec_x) + 1e-6)
            
            # 构建旋转矩阵 (列向量)
            # R_palm = [x, y, z]
            R_palm = np.column_stack((vec_x, vec_y, vec_z))
            
            # 转换为旋转向量供 OpenCV 使用
            r_vec, _ = cv2.Rodrigues(R_palm)
            
            # --- 2. 坐标系转换 (WiLoR Cam -> OpenCV Cam) ---
            # WiLoR 输出的 kpts 已经是 Camera Frame 下的坐标
            # 但是 OpenCV 的 Y轴和Z轴与 WiLoR (OpenGL风格) 是反的
            # 我们需要把 3D 点本身转到 OpenCV 坐标系下再投影
            
            # 定义 3D 轴的端点 (在手掌局部空间)
            # Origin, X-end, Y-end, Z-end
            local_points = np.float32([
                [0, 0, 0],
                [axis_len, 0, 0],
                [0, axis_len, 0],
                [0, 0, axis_len]
            ])
            
            # 将局部点变换到全局 Camera 空间 (WiLoR空间)
            # P_global = R_palm @ P_local + Origin
            # 注意：local_points 是行向量，所以是 P_local @ R_palm.T + Origin
            world_points_wilor = (local_points @ R_palm.T) + origin
            
            # 将 WiLoR 空间点转为 OpenCV 空间点 (Y, Z 取反)
            world_points_opencv = world_points_wilor.copy()
            world_points_opencv[:, 1] *= -1
            world_points_opencv[:, 2] *= -1
            
            # --- 3. 投影绘制 ---
            # 因为我们已经手动转换了点，这里 r_vec 和 t_vec 设为 0 (Identity)
            imgpts, _ = cv2.projectPoints(world_points_opencv, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
            imgpts = imgpts.astype(int).reshape(-1, 2)
            
            # 绘制
            origin_pt = tuple(imgpts[0])
            cv2.line(image, origin_pt, tuple(imgpts[1]), (0, 0, 255), 3) # X - Red (横向/拇指侧)
            cv2.line(image, origin_pt, tuple(imgpts[2]), (0, 255, 0), 3) # Y - Green (纵向/中指)
            cv2.line(image, origin_pt, tuple(imgpts[3]), (255, 0, 0), 3) # Z - Blue (法线)
            
        return image
    
    def _draw_hand_keypoints(self, image: np.ndarray, hands: List[WilorHand]) -> np.ndarray:
        """在单帧图像上绘制手部骨架（关键点和连接线）。"""
        for hand in hands:
            kpts_2d = hand.hand_keypoints_2d.astype(int)
            for p1_idx, p2_idx in MANO_CONNECTIONS:
                p1 = tuple(kpts_2d[p1_idx])
                p2 = tuple(kpts_2d[p2_idx])
                cv2.line(image, p1, p2, (255, 255, 255), 2)
            
            # 颜色映射
            # 0: Wrist(White), 1-4: Thumb(Black), 5-8: Ring(Red), 9-12: Middle(Green), 13-16: Index(Blue), 17-20: Pinky(Grey)
            # 注意：此处遵循用户提供的 connection 索引逻辑
            finger_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (177, 177, 177)]
            joint_to_finger_map = [-1, 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4] # Index mapping based on user provided MANO_CONNECTIONS

            for i, point in enumerate(kpts_2d):
                color = (255, 255, 255) if i == 0 else finger_colors[joint_to_finger_map[i]] if i < len(joint_to_finger_map) else (0,0,0)
                cv2.circle(image, tuple(point), 5, color, -1)
                cv2.circle(image, tuple(point), 5, (0,0,0), 1)
        return image

    def release(self):
        self.cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 WiLoR 模型处理视频并进行3D手部姿态估计。")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument("--save_path", type=str, required=True, help="保存结果的目录路径。")
    args = parser.parse_args()

    wilor_processor = WilorDataset(video_path=args.video_path, save_path=args.save_path)
    wilor_processor.process_and_save_all()
    wilor_processor.release()
    # xvfb-run -a python wilor.py --video_path "./data/mps_open_door_4_vrs/aria/video_original.mp4" --save_path "./data/mps_open_door_4_vrs/wilor/"
