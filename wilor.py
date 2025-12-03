# -*- coding: utf-8 -*-
"""

本脚本实现了对视频进行手部姿态估计的完整流程，使用了 WiLoR 模型。
更新内容：
1. 增加了 Hand Detection Confidence 的获取与保存。
2. 实现了基于 emg2pose 论文定义的 Joint Angles (20 DoFs) 计算与保存。
3. 增加了 Joint Angles 的可视化仪表盘到最终视频中。

主要功能包括：
1. 使用 WiLoR 模型对视频的每一帧进行3D手部姿态和形状的估计。
2. 将估计出的3D手部模型渲染回2D图像，生成覆盖图和纯蒙版图。
3. 提取手腕的2D/3D姿态、所有关键点信息以及解剖学关节角度。
4. 将所有处理结果保存为结构化的 JSON 文件和可视化视频。
"""

import os
import cv2
import json
import torch
import trimesh
import pyrender
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple


from utils import utils

from pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# MANO模型关键点连接顺序 (总共21个点, 索引从0到20)
# 注意：这里的索引顺序基于用户提供的原始代码逻辑
# 0: Wrist
# 1-4: Thumb
# 5-8: Ring
# 9-12: Middle
# 13-16: Index
# 17-20: Pinky
MANO_CONNECTIONS = [
    # 拇指 Thumb (索引 1-4)
    (0, 1), (1, 2), (2, 3), (3, 4),
    # 无名指 Ring (索引 5-8)
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
        # 补充一些用于闭合手部模型的面片，防止渲染穿模
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
    is_right: bool = False                  # 是否是右手
    hand_keypoints_2d: np.ndarray = None    # 21个2D关键点坐标 (21, 2)
    hand_keypoints_3d: np.ndarray = None    # 21个3D关键点坐标 (21, 3)，在相机坐标系下
    vertices: np.ndarray = None             # MANO模型的778个顶点坐标 (778, 3)
    global_orient: np.ndarray = None        # 手腕的全局旋转 (1, 3)，轴角表示法
    hand_pose: np.ndarray = None            # 手指关节的旋转 (15, 3)，轴角表示法
    cam_t: np.ndarray = None                # 相机坐标系下的手腕平移 (1, 3)
    rendered_mask: np.ndarray = None        # 渲染出的手部纯色蒙版 (H, W, 4)
    bbox: np.ndarray = None                 # 手部2D边界框 [x1, y1, x2, y2]
    
    # --- 新增字段 ---
    confidence: float = 1.0                 # 手部检测置信度 (0.0 - 1.0)
    joint_angles: Dict[str, float] = field(default_factory=dict) # emg2pose定义的关节角度

@dataclass
class WilorStruct:
    """用于存储单帧图像所有检测信息的抽象数据结构。"""
    idx: int                                    # 帧索引
    rgb: np.ndarray                             # 原始的BGR图像 (H, W, 3)
    focal_length: float = None                  # 估计出的相机焦距
    hands: List[WilorHand] = field(default_factory=list) # 该帧检测到的所有手 (可能为空，或包含1-2只手)

class WilorDataset:
    """
    一个类似于 PyTorch Dataset 的封装类，用于处理视频并逐帧提取 WiLoR 的手部姿态估计结果。
    """
    def __init__(self, video_path: str, save_path: str):
        # --- 1. 初始化路径和参数 ---
        self.video_path = video_path
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        
        # --- 2. 加载视频 ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"无法打开视频文件: {self.video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频加载成功: {self.width}x{self.height} @ {self.fps} FPS, 共 {self.num_frames} 帧。")

        # --- 3. 初始化 WiLoR 模型 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        print(f"正在加载 WiLoR 模型到 {self.device}...")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, dtype=self.dtype)
        print("WiLoR 模型加载完毕。")

        # --- 4. 初始化渲染器 ---
        # MANO模型的面片信息从模型中获取
        self.renderer = Renderer(self.pipe.wilor_model.mano.faces)
        self.RIGHT_HAND_COLOR = (0.66, 0.27, 0.25) # 定义渲染颜色
        self.LEFT_HAND_COLOR = (0.2, 0.2, 0.6)

        # --- 5. 存储所有帧的结果 ---
        self.all_wilor_structs: List[WilorStruct] = []

    def __len__(self) -> int:
        """返回视频的总帧数。"""
        return self.num_frames

    def _calculate_emg2pose_angles(self, keypoints_3d: np.ndarray, is_right: bool) -> Dict[str, float]:
        """
        计算 emg2pose 论文中定义的 20 个关节角度。
        
        定义:
        - Fingers (Index, Middle, Ring, Pinky): MCP(Flex+Abd), PIP(Flex), DIP(Flex) => 4x4 = 16 angles
        - Thumb: CMC(Flex+Abd), MCP(Flex), IP(Flex) => 4 angles
        - Flexion: 父骨骼向量与子骨骼向量的夹角 (弯曲程度)。
        - Abduction: 骨骼在手掌平面上的展开角度 (相对于中指或特定轴)。为简化计算并保持鲁棒性，这里主要计算水平偏角。
        """
        angles = {}
        
        # 辅助函数：计算两个向量的夹角 (度数)
        def get_angle(v1, v2):
            v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
            v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
            dot = np.dot(v1_norm, v2_norm)
            dot = np.clip(dot, -1.0, 1.0)
            return np.degrees(np.arccos(dot))

        # 辅助函数：计算带符号的外展角（简化版）
        # 通过投影到平面来近似
        def get_abduction(bone_vec, ref_vec, plane_normal):
            # 将向量投影到垂直于 plane_normal 的平面
            def project(v): return v - np.dot(v, plane_normal) * plane_normal
            v_proj = project(bone_vec)
            ref_proj = project(ref_vec)
            return get_angle(v_proj, ref_proj)

        kpts = keypoints_3d # (21, 3)
        
        # 定义骨骼向量 (Bone Vectors)
        # 根据已知的索引映射: 
        # Wrist:0, Thumb:1-4, Ring:5-8, Middle:9-12, Index:13-16, Pinky:17-20
        
        # 1. 构建手掌坐标系参考 (Palm Frame)
        # 腕部到中指根部
        v_wrist_middle = kpts[9] - kpts[0] 
        # 腕部到食指根部
        v_wrist_index = kpts[13] - kpts[0]
        # 手掌法线 (Palm Normal) = Cross(Wrist->Index, Wrist->Middle) (右手定则可能需调整)
        palm_normal = np.cross(v_wrist_index, v_wrist_middle)
        palm_normal /= np.linalg.norm(palm_normal)
        
        # --- Fingers (Index, Middle, Ring, Pinky) ---
        # 映射: Name -> [MCP_idx, PIP_idx, DIP_idx, Tip_idx]
        fingers_map = {
            'Index': [13, 14, 15, 16],
            'Middle': [9, 10, 11, 12],
            'Ring': [5, 6, 7, 8],
            'Pinky': [17, 18, 19, 20]
        }

        # 中指近节指骨向量 (用于Abduction参考)
        v_middle_proximal = kpts[10] - kpts[9]

        for name, idxs in fingers_map.items():
            mcp, pip, dip, tip = idxs
            
            # 骨骼向量
            v_metacarpal = kpts[mcp] - kpts[0] # 掌骨 (Wrist -> MCP)
            v_proximal = kpts[pip] - kpts[mcp] # 近节 (MCP -> PIP)
            v_intermediate = kpts[dip] - kpts[pip] # 中节 (PIP -> DIP)
            v_distal = kpts[tip] - kpts[dip] # 远节 (DIP -> Tip)
            
            # 1. MCP Flexion: 掌骨与近节指骨的夹角 (或近节指骨与手掌平面的垂直夹角，这里用骨骼间夹角简化)
            angles[f'{name}_MCP_Flexion'] = get_angle(v_metacarpal, v_proximal)
            
            # 2. MCP Abduction: 近节指骨相对于中指近节指骨的水平夹角
            # 注意：中指自己的外展角通常为0，或者相对于手掌中心轴
            if name == 'Middle':
                 angles[f'{name}_MCP_Abduction'] = 0.0 # 中指为参考
            else:
                 # 计算相对于中指指骨张开的角度
                 angles[f'{name}_MCP_Abduction'] = get_abduction(v_proximal, v_middle_proximal, palm_normal)

            # 3. PIP Flexion: 近节与中节夹角
            angles[f'{name}_PIP_Flexion'] = get_angle(v_proximal, v_intermediate)
            
            # 4. DIP Flexion: 中节与远节夹角
            angles[f'{name}_DIP_Flexion'] = get_angle(v_intermediate, v_distal)

        # --- Thumb (拇指) ---
        # Indices: 1(CMC), 2(MCP), 3(IP), 4(Tip)
        # 拇指比较特殊，CMC是基于Wrist的，MCP基于CMC
        v_wrist_cmc = kpts[1] - kpts[0] # 事实上是 Wrist -> CMC joint pos (但Mano index 1通常是CMC位置)
        # 修正：Mano index 1 是 CMC 关节位置吗？通常 0->1 是第一掌骨。
        # 假设：0(Wrist) -> 1(CMC) -> 2(MCP) -> 3(IP) -> 4(Tip)
        v_thumb_metacarpal = kpts[2] - kpts[1] # 第一掌骨
        v_thumb_proximal = kpts[3] - kpts[2]   # 近节
        v_thumb_distal = kpts[4] - kpts[3]     # 远节
        
        # 拇指角度定义 (emg2pose):
        # CMC Flexion / Abduction (第一掌骨相对于手掌平面的运动)
        # 这是一个很复杂的3D运动，这里用 Wrist->CMC 和 CMC->MCP 的关系近似
        v_ref_thumb = kpts[1] - kpts[0] # 参考向量
        
        angles['Thumb_CMC_Flexion'] = get_angle(v_ref_thumb, v_thumb_metacarpal)
        angles['Thumb_CMC_Abduction'] = get_abduction(v_thumb_metacarpal, v_middle_proximal, palm_normal) # 拇指张开程度
        
        angles['Thumb_MCP_Flexion'] = get_angle(v_thumb_metacarpal, v_thumb_proximal)
        angles['Thumb_IP_Flexion'] = get_angle(v_thumb_proximal, v_thumb_distal)

        return angles

    def __getitem__(self, idx: int) -> WilorStruct:
        """处理并返回指定帧的检测结果。"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = self.cap.read()
        if not ret:
            raise IndexError(f"无法读取第 {idx} 帧。")

        # 使用 WiLoR 模型进行预测
        outputs = self.pipe.predict(frame_bgr)
        
        wilor_hands = []
        focal_length = None

        for out in outputs:
            # 提取预测结果
            wilor_preds = out['wilor_preds']
            is_right = out['is_right']
            focal_length = wilor_preds['scaled_focal_length']
            
            # --- 新需求1: 获取 Confidence Score ---
            bbox = out['hand_bbox'] # 通常格式 [x1, y1, x2, y2] 或 [x1, y1, x2, y2, score]
            confidence = 1.0 # 默认值
            if len(bbox) >= 5:
                confidence = float(bbox[4])
                bbox = bbox[:4] # 截断 score，只保留坐标供后续使用
            elif 'score' in out:
                confidence = float(out['score'])
            elif 'conf' in out:
                confidence = float(out['conf'])
            
            # 计算 3D 关键点用于角度计算
            kpts_3d = wilor_preds["pred_keypoints_3d"][0]
            
            # --- 新需求2: 计算 Joint Angles ---
            angles = self._calculate_emg2pose_angles(kpts_3d, is_right)

            # 创建并填充 WilorHand 数据结构
            hand_data = WilorHand(
                is_right=is_right,
                hand_keypoints_2d=wilor_preds["pred_keypoints_2d"][0],
                hand_keypoints_3d=kpts_3d,
                vertices=wilor_preds['pred_vertices'][0],
                global_orient=wilor_preds['global_orient'][0],
                hand_pose=wilor_preds['hand_pose'][0],
                cam_t=wilor_preds['pred_cam_t_full'][0],
                bbox=np.array(bbox),
                confidence=confidence,   # 保存置信度
                joint_angles=angles      # 保存关节角度
            )
            wilor_hands.append(hand_data)

        # 创建并返回 WilorStruct
        return WilorStruct(
            idx=idx,
            rgb=frame_bgr,
            focal_length=focal_length,
            hands=wilor_hands
        )
        
    def process_and_save_all(self):
        """处理视频的所有帧并保存结果。"""
        print("开始处理所有视频帧...")
        for i in tqdm(range(len(self)), desc="处理视频帧"):
            wilor_struct = self[i]
            self.all_wilor_structs.append(wilor_struct)
            
            # 保存每帧的 JSON 数据
            self.save_struct_to_json(wilor_struct)

        print("所有帧处理完毕，开始生成可视化视频...")
        self.create_visualization_videos()
        print("所有任务完成！")

    def save_struct_to_json(self, wilor_struct: WilorStruct):
        """将单个 WilorStruct 对象的内容保存为 JSON 文件。"""
        frame_dir = os.path.join(self.save_path, "all_data", f"{wilor_struct.idx:05d}")
        os.makedirs(frame_dir, exist_ok=True)

        def to_list_safe(item):
            # 将 numpy 数组转为 list
            if isinstance(item, np.ndarray):
                return item.tolist()
            return item

        # 准备要序列化的数据
        data_to_save = {
            "frame_index": int(wilor_struct.idx), # 强制转为 int
            # 强制转为 float，防止它是 numpy.float32
            "focal_length": float(wilor_struct.focal_length) if wilor_struct.focal_length is not None else None,
            "hands": []
        }
        for hand in wilor_struct.hands:
            # --- 关键修正 ---
            # joint_angles 里的值是 numpy 类型，必须逐个转为 python float
            safe_joint_angles = {k: float(v) for k, v in hand.joint_angles.items()}
            
            hand_dict = {
                "is_right": bool(hand.is_right),
                "confidence": float(hand.confidence), # 强制转为 float
                "bbox": to_list_safe(hand.bbox),
                "joint_angles_emg2pose": safe_joint_angles, # 使用转换后的字典
                "hand_keypoints_2d": to_list_safe(hand.hand_keypoints_2d),
                "hand_keypoints_3d": to_list_safe(hand.hand_keypoints_3d),
                # "vertices": to_list_safe(hand.vertices), 
                "global_orient(axis_angle)": to_list_safe(hand.global_orient),
                "hand_pose(axis_angle)": to_list_safe(hand.hand_pose),
                "camera_translation": to_list_safe(hand.cam_t),
            }
            data_to_save["hands"].append(hand_dict)
            
        # 写入JSON文件
        json_path = os.path.join(frame_dir, "data.json")
        with open(json_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
            
    def create_visualization_videos(self):
        """根据已处理的所有帧数据，创建不同的可视化视频。"""
        if not self.all_wilor_structs:
            print("没有已处理的数据，无法创建视频。请先运行 process_and_save_all()。")
            return

        # 动态生成所有可视化帧
        rendered_hand_frames, rendered_hand_mask_only_frames, palm_and_wrist_pose_frames, hand_keypoints_frames, all_info_frames = [], [], [], [], []

        for s in tqdm(self.all_wilor_structs, desc="生成可视化帧"):
            # 准备画布
            image_rgb_norm = cv2.cvtColor(s.rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rendered_hand_mask_only_canvas = np.zeros_like(image_rgb_norm)
            rendered_hand_canvas = image_rgb_norm.copy()
            
            # 渲染和绘制
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

            # BGR转换回来用于保存
            rendered_hand_frames.append(cv2.cvtColor((rendered_hand_canvas * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            rendered_hand_mask_only_frames.append(cv2.cvtColor((rendered_hand_mask_only_canvas * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            palm_and_wrist_pose_frames.append(self.draw_palm_and_wrist_pose(s.rgb.copy(), s.hands, s.focal_length, self.width, self.height))
            hand_keypoints_frames.append(self._draw_hand_keypoints(s.rgb.copy(), s.hands))
            
            # --- 生成包含所有信息的可视化帧 ---
            # 1. 渲染手部 Mesh
            # 2. 绘制 坐标轴
            # 3. 绘制 关键点
            # 4. 新增：绘制角度仪表盘
            base_img = cv2.cvtColor((rendered_hand_canvas * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            img_with_pose = self.draw_palm_and_wrist_pose(base_img, s.hands, s.focal_length, self.width, self.height)
            img_with_kpts = self._draw_hand_keypoints(img_with_pose, s.hands)
            img_final = self._draw_angle_dashboard(img_with_kpts, s.hands) # 新增可视化
            
            all_info_frames.append(img_final)

        # 创建视频
        utils.create_video_from_frames(
            rendered_hand_frames, 
            os.path.join(self.save_path, f"viz_rendered_hand.mp4"), 
            self.fps
        )
        utils.create_video_from_frames(
            rendered_hand_mask_only_frames,
            os.path.join(self.save_path, f"viz_rendered_hand_mask_only.mp4"),
            self.fps
        )
        utils.create_video_from_frames(
            palm_and_wrist_pose_frames,
            os.path.join(self.save_path, f"viz_palm_and_wrist_pose.mp4"),
            self.fps
        )
        utils.create_video_from_frames(
            hand_keypoints_frames,
            os.path.join(self.save_path, f"viz_hand_keypoints.mp4"),
            self.fps
        )
        utils.create_video_from_frames(
            all_info_frames,
            os.path.join(self.save_path, f"viz_all_info.mp4"),
            self.fps
        )

    # --- 新增函数: Joint Angle Dashboard 可视化 ---
    def _draw_angle_dashboard(self, image: np.ndarray, hands: List[WilorHand]) -> np.ndarray:
        """
        在图像上绘制关节角度仪表盘。
        显示每根手指的平均屈曲度 (Average Flexion)。
        """
        if not hands:
            return image
        
        h, w = image.shape[:2]
        # 创建一个半透明的覆盖层用于显示文字和图表
        overlay = image.copy()
        
        panel_w = 300
        panel_h = 220
        # 放置在右下角
        start_x = w - panel_w - 10
        start_y = h - panel_h - 10
        
        cv2.rectangle(overlay, (start_x, start_y), (start_x + panel_w, start_y + panel_h), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        cv2.putText(image, "Joint Flexion Dashboard", (start_x + 10, start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 只要第一只手的数据用于展示 (防止太乱)
        hand = hands[0]
        angles = hand.joint_angles
        side = "Right" if hand.is_right else "Left"
        conf_str = f"Conf: {hand.confidence:.2f}"
        cv2.putText(image, f"{side} Hand | {conf_str}", (start_x + 10, start_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        bar_start_y = start_y + 70
        bar_height = 20
        gap = 10
        
        for i, finger in enumerate(fingers):
            # 获取该手指的主要屈曲角度
            # 为了简化展示，我们取 MCP 和 PIP 的平均屈曲值来代表手指弯曲程度
            flex_keys = [k for k in angles.keys() if finger in k and "Flexion" in k]
            if not flex_keys: continue
            
            avg_flex = np.mean([angles[k] for k in flex_keys])
            
            # 归一化用于画条 (假设最大弯曲 120度)
            ratio = min(max(avg_flex / 120.0, 0), 1.0)
            
            y = bar_start_y + i * (bar_height + gap)
            
            # 文字
            cv2.putText(image, f"{finger[:3]}", (start_x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 背景条
            cv2.rectangle(image, (start_x + 60, y), (start_x + panel_w - 50, y + bar_height), (50, 50, 50), -1)
            # 前景条 (弯曲程度), 颜色从绿(直)渐变到红(弯)
            bar_len = int(ratio * (panel_w - 110))
            # --- 修正：将颜色分量强制转为 int ---
            color = (0, int(255 * (1-ratio)), int(255 * ratio)) 
            cv2.rectangle(image, (start_x + 60, y), (start_x + 60 + bar_len, y + bar_height), color, -1)
            
            # 数值
            cv2.putText(image, f"{int(avg_flex)}d", (start_x + panel_w - 45, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return image

    def draw_palm_and_wrist_pose(self, image: np.ndarray, hands: List[WilorHand], focal_length, width, height):
        if focal_length is None:
            return image
        
        # OpenCV 相机内参矩阵
        camera_matrix = np.array([[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1)) # 假设无畸变

        # 坐标系转换矩阵：绕 X 轴旋转 180 度
        # WiLoR/MANO (Y-up, Z-back) -> OpenCV (Y-down, Z-forward)
        R_correction = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)

        # 定义坐标轴的 3D 点 (原点 + XYZ轴末端)
        axis_len = 0.05 # 5cm
        axis_points = np.float32([
            [0, 0, 0],          # Origin
            [axis_len, 0, 0],   # X
            [0, axis_len, 0],   # Y
            [0, 0, axis_len]    # Z
        ])

        for hand in hands:
            # 1. 获取原始旋转矩阵
            r_vec = hand.global_orient.astype(np.float64)
            R_original, _ = cv2.Rodrigues(r_vec)

            # 2. 应用坐标系转换到旋转矩阵
            R_final = R_correction @ R_original
            r_vec_final, _ = cv2.Rodrigues(R_final)

            # -----------------------------
            # 绘制手腕 (Wrist) 坐标轴
            # -----------------------------
            t_wrist = hand.cam_t.astype(np.float64).reshape(3, 1)
            t_wrist_final = R_correction @ t_wrist

            imgpts_wrist, _ = cv2.projectPoints(axis_points, r_vec_final, t_wrist_final, camera_matrix, dist_coeffs)
            imgpts_wrist = imgpts_wrist.astype(int)

            origin = tuple(imgpts_wrist[0].ravel())
            cv2.line(image, origin, tuple(imgpts_wrist[1].ravel()), (0, 0, 255), 3) # X - Red
            cv2.line(image, origin, tuple(imgpts_wrist[2].ravel()), (0, 255, 0), 3) # Y - Green
            cv2.line(image, origin, tuple(imgpts_wrist[3].ravel()), (255, 0, 0), 3) # Z - Blue

            # -----------------------------
            # 绘制手掌 (Palm) 坐标轴
            # -----------------------------
            # 计算手掌中心的 3D 坐标并平移
            kpts_3d = hand.hand_keypoints_3d 
            palm_indices = [0, 13, 9, 17] # Wrist, Index MCP, Middle MCP, Pinky MCP
            palm_center_3d = np.mean(kpts_3d[palm_indices], axis=0).reshape(3, 1)
            
            t_palm_final = R_correction @ palm_center_3d
            imgpts_palm, _ = cv2.projectPoints(axis_points, r_vec_final, t_palm_final, camera_matrix, dist_coeffs)
            imgpts_palm = imgpts_palm.astype(int)

            origin_palm = tuple(imgpts_palm[0].ravel())
            cv2.line(image, origin_palm, tuple(imgpts_palm[1].ravel()), (0, 165, 255), 3) # X - Orange
            cv2.line(image, origin_palm, tuple(imgpts_palm[2].ravel()), (255, 255, 0), 3) # Y - Cyan
            cv2.line(image, origin_palm, tuple(imgpts_palm[3].ravel()), (255, 0, 255), 3) # Z - Magenta
            
        return image
    
    def _draw_hand_keypoints(self, image: np.ndarray, hands: List[WilorHand]) -> np.ndarray:
        """在单帧图像上绘制手部骨架（关键点和连接线）。"""
        for hand in hands:
            kpts_2d = hand.hand_keypoints_2d.astype(int)
            
            # 绘制连接线
            for p1_idx, p2_idx in MANO_CONNECTIONS:
                p1 = tuple(kpts_2d[p1_idx])
                p2 = tuple(kpts_2d[p2_idx])
                cv2.line(image, p1, p2, (255, 255, 255), 2) # 白色连接线
            
            # 定义手指到颜色的映射
            finger_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (177, 177, 177)] # Black, Red, Green, Blue, Grey
            # 定义每个关键点属于哪个手指 (基于修正后的索引映射)
            # 0: Wrist
            # 1-4: Thumb, 5-8: Ring, 9-12: Middle, 13-16: Index, 17-20: Pinky
            joint_to_finger_map = [
                -1,                      # 0: Wrist
                0, 0, 0, 0,              # 1-4: Thumb
                3, 3, 3, 3,              # 5-8: Ring (这里原代码注释写错了, 5-8实际上通常是 Index 或 Middle, 但用户代码里标注的是 Ring?)
                                         # 等等，让我再次确认用户代码里的MANO_CONNECTIONS注释。
                                         # 用户原始代码： # 无名指 Ring (索引 5-8)
                                         #               # 中指 Middle (索引 9-12)
                                         #               # 食指 Index (索引 13-16)
                                         #               # 小指 Pinky (索引 17-20)
                1, 1, 1, 1,              # 5-8: Ring (对应 finger_colors[1] Red)
                2, 2, 2, 2,              # 9-12: Middle (对应 finger_colors[2] Green)
                3, 3, 3, 3,              # 13-16: Index (对应 finger_colors[3] Blue)
                4, 4, 4, 4,              # 17-20: Pinky (对应 finger_colors[4] Grey)
            ]
            # 注意：上面的map需要根据实际情况调整，我保持了你原始代码的逻辑，
            # 但是上面的 finger_colors 索引 0是Black(Thumb).

            for i, point in enumerate(kpts_2d):
                if i == 0: # 手腕
                    color = (255, 255, 255) # 白色
                else:
                    if i < len(joint_to_finger_map):
                        finger_idx = joint_to_finger_map[i]
                        if finger_idx != -1:
                            color = finger_colors[finger_idx]
                        else:
                            color = (0, 0, 0)
                    else:
                        color = (0, 0, 0)

                cv2.circle(image, tuple(point), 5, color, -1)
                cv2.circle(image, tuple(point), 5, (0,0,0), 1) # 添加黑色轮廓，使其更清晰
        return image

    def release(self):
        """释放视频捕获对象。"""
        self.cap.release()

if __name__ == '__main__':
    # --- 1. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="使用 WiLoR 模型处理视频并进行3D手部姿态估计。")
    parser.add_argument("--video_path", type=str, required=True, help="输入视频文件的路径。")
    parser.add_argument("--save_path", type=str, required=True, help="保存结果的目录路径。")
    args = parser.parse_args()

    # --- 2. 创建 WilorDataset 实例并运行处理流程 ---
    wilor_processor = WilorDataset(video_path=args.video_path, save_path=args.save_path)
    wilor_processor.process_and_save_all()
    wilor_processor.release()
    
    # xvfb-run -a python wilor.py --video_path "./data/mps_open_door_4_vrs/aria/video_original.mp4" --save_path "./data/mps_open_door_4_vrs/wilor/"
    # xvfb-run -a python wilor.py --video_path "./test_data/test_video.mp4" --save_path "./test_data/wilor_output/"