#!/usr/bin/env python3
"""分析 motion.onnx 模型的输入/输出维度"""

import onnxruntime as ort
import numpy as np
import os

# 模型路径
model_path = "/home/meme/Documents/unitree_h1/unitree_rl_gym/deploy/pre_train/h1_2/motion.onnx"

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在: {model_path}")
    exit(1)

# 加载 ONNX 模型
session = ort.InferenceSession(model_path)

# 获取输入信息
print("=" * 60)
print("ONNX 模型输入信息:")
print("=" * 60)
for i, input_info in enumerate(session.get_inputs()):
    name = input_info.name
    shape = input_info.shape
    type_str = input_info.type
    print(f"输入 {i}:")
    print(f"  名称: {name}")
    print(f"  形状: {shape}")
    print(f"  类型: {type_str}")
    
    # 计算实际维度（去除 batch 维度）
    if len(shape) == 2:
        actual_dim = shape[1]
        print(f"  实际观测维度: {actual_dim}")

# 获取输出信息
print("\n" + "=" * 60)
print("ONNX 模型输出信息:")
print("=" * 60)
for i, output_info in enumerate(session.get_outputs()):
    name = output_info.name
    shape = output_info.shape
    type_str = output_info.type
    print(f"输出 {i}:")
    print(f"  名称: {name}")
    print(f"  形状: {shape}")
    print(f"  类型: {type_str}")
    
    # 计算实际维度（去除 batch 维度）
    if len(shape) == 2:
        actual_dim = shape[1]
        print(f"  实际动作维度: {actual_dim}")

# 测试推理
print("\n" + "=" * 60)
print("测试推理:")
print("=" * 60)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# 创建随机输入
if len(input_shape) == 2:
    batch_size = 1
    obs_dim = input_shape[1]
    # 处理动态维度
    if isinstance(obs_dim, str):
        # 从 h1_2.yaml 获取实际维度
        obs_dim = 47
    test_input = np.random.randn(batch_size, obs_dim).astype(np.float32)
    print(f"创建测试输入: shape={test_input.shape}")
    
    # 运行推理
    output = session.run(None, {input_name: test_input})
    print(f"推理输出: shape={output[0].shape}")
    print(f"输出值范围: min={output[0].min():.4f}, max={output[0].max():.4f}")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)