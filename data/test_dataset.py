# -*- coding: utf-8 -*-
# @Time : 13:54 2024/3/14
# @Author : Allen-wan
# @File : test_dataset.py
# @Software : PyCharm
import os
import torch

# 假设 _CASIA_OUMVLP 和 ImprovedDataSet 已经正确定义和实现
from DataReader import DataReader  # 替换为包含函数的实际模块名

def test_dataset():
    # 设置参数（按照您的实际情况调整这些参数）
    batch_size = [8, 16]
    test_batch_size = 1
    num_workers = 0  # 对于测试，通常设置为0
    dataset_path = '/home/user/Data/CASIA-B/cut'  # 更改为您的数据路径
    list_path = '/home/user/Data/CASIA-B/list'  # 更改为您的列表路径
    frame_num = 30
    resolution = 64
    pid_num = 73  # 可以根据您的数据集进行调整
    pid_shuffle = False

    # 获取训练和测试加载器
    train_loader, test_loader = DataReader(batch_size, test_batch_size, num_workers, dataset_path, list_path, frame_num, resolution, pid_num, pid_shuffle)

    # 测试训练数据加载器
    print("Testing train loader...")
    for i, (data, labels, seq_types, views) in enumerate(train_loader):
        print(f"Batch {i+1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels: {labels}")
        print(f"Sequence types: {seq_types}")
        print(f"Views: {views}")
        if i == 1:  # 只测试前两个批次
            break

    # 测试测试数据加载器
    print("\nTesting test loader...")
    for i, (data, labels, seq_types, views) in enumerate(test_loader):
        print(f"Batch {i+1}")
        print(f"Data shape: {data.shape}")
        print(f"Labels: {labels}")
        print(f"Sequence types: {seq_types}")
        print(f"Views: {views}")
        if i == 1:  # 只测试前两个批次
            break

if __name__ == "__main__":
    test_dataset()
