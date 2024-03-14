# -*- coding: utf-8 -*-
# @Time : 10:59 2024/3/13
# @Author : Allen-wan
# @File : DataReader.py
# @Software : PyCharm
import os
import numpy as np
import torch
import torch.utils.data as tordata
from pathlib import Path
from ImprovedTripletSampler import ImprovedTripletSampler
from GaitDataset import OptimizedGaitDataset  # 假定您已经创建了改进后的数据集类


def DataReader(batch_size=[8, 16], test_batch_size=1, num_workers=3, dataset_path='./data/CASIA/processed/',
                  list_path='./data/CASIA/list/', frame_num=30, resolution=64, pid_num=73, pid_shuffle=False):
    """
    Construct CASIA's trainset and testset.
    """
    list_path = Path(list_path)
    dataset_path = Path(dataset_path)
    list_path.mkdir(parents=True, exist_ok=True)  # 创建list目录，如果不存在

    # 构造或获取数据列表
    train_list_path = list_path / f'{pid_num}_{pid_shuffle}_train_list_seq.npy'
    test_list_path = list_path / f'{pid_num}_{pid_shuffle}_test_list_seq.npy'

    if train_list_path.exists() and test_list_path.exists():
        train_list = np.load(train_list_path, allow_pickle=True)
        test_list = np.load(test_list_path, allow_pickle=True)
    else:
        train_list, test_list = generate_data_lists(dataset_path, pid_num, pid_shuffle)
        np.save(train_list_path, train_list)
        np.save(test_list_path, test_list)

    # 创建数据集实例
    train_source = OptimizedGaitDataset(train_list[0], train_list[1], train_list[2], train_list[3], resolution, frame_num)
    test_source = OptimizedGaitDataset(test_list[0], test_list[1], test_list[2], test_list[3], resolution, -1)  # 使用所有帧进行测试

    # 构建数据加载器
    train_sampler = ImprovedTripletSampler(train_source, batch_size)
    train_loader = tordata.DataLoader(train_source, batch_sampler=train_sampler, num_workers=num_workers,
                                      pin_memory=True)
    test_loader = tordata.DataLoader(test_source, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def generate_data_lists(dataset_path, pid_num, pid_shuffle):
    """
    Generates training and testing lists.
    """
    dataset_path = Path(dataset_path)
    seq_dirs, labels, seq_types, views = [], [], [], []

    # 遍历数据集目录以收集数据
    for label_dir in sorted(dataset_path.iterdir()):
        if label_dir.is_dir() and (not 'CASIA' in str(dataset_path) or label_dir.name != '005'):  # 排除特定情况
            for seq_type_dir in sorted(label_dir.iterdir()):
                for view_dir in sorted(seq_type_dir.iterdir()):
                    if len(list(view_dir.iterdir())) > 15:  # 确保序列长度足够
                        seq_dirs.append(str(view_dir))
                        labels.append(label_dir.name)
                        seq_types.append(seq_type_dir.name)
                        views.append(view_dir.name)

    # 创建ID列表和训练/测试分割
    pid_list = sorted(set(labels))
    if pid_shuffle:
        np.random.shuffle(pid_list)
    train_pids, test_pids = pid_list[:pid_num], pid_list[pid_num:]

    # 根据ID列表分配训练和测试数据
    train_list = [[], [], [], []]
    test_list = [[], [], [], []]
    for dir, label, seq_type, view in zip(seq_dirs, labels, seq_types, views):
        if label in train_pids:
            for lst, item in zip(train_list, [dir, label, seq_type, view]):
                lst.append(item)
        elif label in test_pids:
            for lst, item in zip(test_list, [dir, label, seq_type, view]):
                lst.append(item)

    return np.array(train_list), np.array(test_list)

