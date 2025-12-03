Joint Angles (关节角度) 计算 (参考 emg2pose)：
定义解释：根据 emg2pose 论文，他们关注的是符合解剖学的角度（共20个自由度）：
手指 (食指、中指、无名指、小指)：每个手指4个自由度。
MCP (掌指关节)：包含 Flexion (屈曲) 和 Abduction (外展/张开)。
PIP (近端指间关节)：仅 Flexion。
DIP (远端指间关节)：仅 Flexion。
拇指：4个自由度。
CMC (腕掌关节)：包含 Flexion 和 Abduction。
MCP：仅 Flexion。
IP：仅 Flexion。
实现方式：在代码中新增了 _calculate_emg2pose_angles 函数。它利用 3D 关键点向量计算骨骼间的夹角。
屈曲 (Flexion)：计算父骨骼向量与子骨骼向量之间的夹角。
外展 (Abduction)：计算手指根部骨骼相对于中指（参考轴）的水平偏离角。


---

### 第一部分：Joint Angle 的定义 (物理意义)

在 *emg2pose* 这篇论文（以及大多数手部生物力学模型，如 MANO）中，手部姿态被描述为一系列**关节的旋转角度**。

人手通常被建模为 **20 个自由度 (Degrees of Freedom, DoF)** 的关节角度集合。

#### 1. 四根手指 (食指、中指、无名指、小指)
每根手指有 3 个关节，共贡献 **4 个角度**：

*   **MCP 关节 (掌指关节/指根关节)**：手指连接手掌的地方。
    *   **Flexion (屈曲)**: 手指向下弯曲的角度（握拳动作）。
    *   **Abduction (外展)**: 手指左右张开的角度（五指张开动作）。
*   **PIP 关节 (近端指间关节/中间关节)**：
    *   **Flexion (屈曲)**: 中间关节的弯曲。它只能弯曲，不能左右摆动。
*   **DIP 关节 (远端指间关节/指尖关节)**：
    *   **Flexion (屈曲)**: 指尖关节的弯曲。同样只能弯曲。

**计算：** 4根手指 $\times$ 4个角度 = **16个角度**。

#### 2. 拇指 (Thumb)
拇指的结构比较特殊，它也有 **4 个角度**：

*   **CMC 关节 (腕掌关节/拇指根部)**：深埋在手腕处，是鞍状关节，非常灵活。
    *   **Flexion (屈曲)**: 拇指向手心方向运动。
    *   **Abduction (外展)**: 拇指垂直于手掌平面立起来的运动。
*   **MCP 关节 (掌指关节)**：
    *   **Flexion (屈曲)**: 拇指中间关节的弯曲。
*   **IP 关节 (指间关节)**：
    *   **Flexion (屈曲)**: 拇指尖端的弯曲。

**计算：** 拇指 $\times$ 4个角度 = **4个角度**。

**总计：16 + 4 = 20 个 Joint Angles。**

---

### 第二部分：结合代码解释如何获得 (数学实现)

在 WiLoR 中，我们获得的是 **3D Keypoints (x, y, z 坐标)**。要把坐标转换成角度，本质上是计算**向量之间的夹角**。

请看代码中的 `_calculate_emg2pose_angles` 函数，核心逻辑如下：

#### 1. 核心数学工具：点积公式 (Dot Product)
计算两个向量 $\vec{a}$ 和 $\vec{b}$ 之间的夹角 $\theta$：
$$ \vec{a} \cdot \vec{b} = |\vec{a}| |\vec{b}| \cos(\theta) $$
$$ \theta = \arccos\left( \frac{\vec{a} \cdot \vec{b}}{|\vec{a}| |\vec{b}|} \right) $$

代码对应：
```python
def get_angle(v1, v2):
    # 归一化向量 (变成长度为1的单位向量)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
    # 计算点积
    dot = np.dot(v1_norm, v2_norm)
    # 限制范围防止计算误差导致报错 (-1 到 1)
    dot = np.clip(dot, -1.0, 1.0)
    # 反余弦得到弧度，转为角度
    return np.degrees(np.arccos(dot))
```

#### 2. 计算屈曲 (Flexion) - 以食指为例
屈曲就是“父骨骼”和“子骨骼”之间的弯曲程度。

*   **Keypoints 索引**:
    *   0: Wrist (手腕)
    *   13: MCP (指根)
    *   14: PIP (中间)
    *   15: DIP (指尖前)
    *   16: Tip (指尖)

*   **构建骨骼向量**:
    *   `v_metacarpal` (掌骨): 0 -> 13
    *   `v_proximal` (近节指骨): 13 -> 14
    *   `v_intermediate` (中节指骨): 14 -> 15
    *   `v_distal` (远节指骨): 15 -> 16

*   **计算角度**:
    *   **MCP Flexion**: 计算 `v_metacarpal` 和 `v_proximal` 的夹角。
    *   **PIP Flexion**: 计算 `v_proximal` 和 `v_intermediate` 的夹角。
    *   **DIP Flexion**: 计算 `v_intermediate` 和 `v_distal` 的夹角。

代码对应：
```python
# 1. MCP Flexion: 掌骨与近节指骨的夹角
angles[f'{name}_MCP_Flexion'] = get_angle(v_metacarpal, v_proximal)

# 3. PIP Flexion: 近节与中节夹角
angles[f'{name}_PIP_Flexion'] = get_angle(v_proximal, v_intermediate)

# 4. DIP Flexion: 中节与远节夹角
angles[f'{name}_DIP_Flexion'] = get_angle(v_intermediate, v_distal)
```

#### 3. 计算外展 (Abduction) - 比较难点
外展是指手指在手掌平面上**左右偏离**的角度。要计算这个，我们需要一个“基准线”和一个“平面”。

*   **基准线**: 通常以**中指 (Middle Finger)** 为中心轴。
*   **计算逻辑**:
    1.  计算**手掌法线 (Palm Normal)**：利用手腕、食指根、中指根三个点确定一个平面，计算其垂直向量。
    2.  计算手指相对于中指的偏离。

代码对应：
```python
# 构建手掌法线
v_wrist_middle = kpts[9] - kpts[0]
v_wrist_index = kpts[13] - kpts[0]
palm_normal = np.cross(v_wrist_index, v_wrist_middle) # 叉乘得到法线

# 计算 Abduction
if name == 'Middle':
    angles[f'{name}_MCP_Abduction'] = 0.0 # 中指自己是基准，设为0
else:
    # 计算当前手指(v_proximal) 相对于 中指(v_middle_proximal) 的水平夹角
    # get_abduction 内部会将向量投影到手掌平面上，排除弯曲的影响，只看左右偏离
    angles[f'{name}_MCP_Abduction'] = get_abduction(v_proximal, v_middle_proximal, palm_normal)
```

#### 4. 拇指的特殊处理
拇指的 `CMC Flexion` 和 `Abduction` 定义比较复杂，因为拇指是斜着长的。
代码中采用了简化的近似：
*   **CMC Flexion**: 第一掌骨（手腕到拇指根）相对于参考向量的弯曲。
*   **CMC Abduction**: 第一掌骨相对于手掌平面的张开程度。

---

### 总结：你需要保存的数据

学长想要的 `joint angle` 就是这个包含 20 个浮点数的字典。

在生成的 `data.json` 中，结构会是这样（你现在的代码已经生成了这个）：

```json
"joint_angles_emg2pose": {
    "Index_MCP_Flexion": 10.5,
    "Index_MCP_Abduction": 2.1,
    "Index_PIP_Flexion": 45.0,
    "Index_DIP_Flexion": 20.3,
    "Middle_MCP_Flexion": ...
    ... (共20个)
}
```

这些数据可以直接用于训练 emg2pose 这样的模型，或者驱动机器人手/虚拟形象的手。