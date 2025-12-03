# aria keypoints:
0: thumb tip
1: index tip
2: middle tip
3: ring tip
4: pinky tip
5,6,7: other thumb keypoints (the order: from wrist to tip, 5 is the wrist point)
8,9,10: other index keypoints (the order: from wrist to tip)
11,12,13: other middle keypoints (the order: from wrist to tip)
14,15,16: other ring keypoints (the order: from wrist to tip)
17,18,19: other pinky keypoints (the order: from wrist to tip)
20: palm point

<!-- #
Name: THUMB_FINGERTIP           | Index: 0
Name: INDEX_FINGERTIP           | Index: 1
Name: MIDDLE_FINGERTIP          | Index: 2
Name: RING_FINGERTIP            | Index: 3
Name: PINKY_FINGERTIP           | Index: 4
Name: WRIST                     | Index: 5
Name: THUMB_INTERMEDIATE        | Index: 6
Name: THUMB_DISTAL              | Index: 7
Name: INDEX_PROXIMAL            | Index: 8
Name: INDEX_INTERMEDIATE        | Index: 9
Name: INDEX_DISTAL              | Index: 10
Name: MIDDLE_PROXIMAL           | Index: 11
Name: MIDDLE_INTERMEDIATE       | Index: 12
Name: MIDDLE_DISTAL             | Index: 13
Name: RING_PROXIMAL             | Index: 14
Name: RING_INTERMEDIATE         | Index: 15
Name: RING_DISTAL               | Index: 16
Name: PINKY_PROXIMAL            | Index: 17
Name: PINKY_INTERMEDIATE        | Index: 18
Name: PINKY_DISTAL              | Index: 19
Name: PALM_CENTER               | Index: 20
Name: NUM_LANDMARKS             | Index: 21
# -->


# wilor keypoints:
0: wrist point
1,2,3,4: thumb (the order: from wrist to tip)
5,6,7,8: index (the order: from wrist to tip)
9,10,11,12: middle (the order: from wrist to tip)
13,14,15,16: ring (the order: from wrist to tip)
17,18,19,20: pinky (the order: from wrist to tip)

我搞清楚他们keypoints的顺序了，xy坐标轴没有区别，请你帮我写一个脚本（参考egozero那篇）融合这两个数据，把每一帧的数据（aria原来有的数据）都重新保存到./data/mps_grasp_phone_vrs/aria_and_wilor/all_data. 生成并且可视化keypoints和wrist_pose