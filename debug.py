# -*- coding: utf-8 -*-
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import os
import hydra

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')

import torch.distributed as dist
from torch.utils.data import DataLoader
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService

# rank = int(os.environ.get('RANK', '0'))
# local_rank = int(os.environ.get('LOCAL_RANK', '0'))
# world_size = int(os.environ.get('WORLD_SIZE', '1'))
# def initialize_distributed(rank, local_rank, world_size, backend='nccl'):
#     dist.init_process_group(
#         backend=backend,
#         world_size=world_size,
#         rank=0
#     )

# # Initialize the process group
# initialize_distributed(rank, local_rank, world_size, backend='hccl')

# dataset_cfg = "configs/data/sdxl_adapter_finetune.yaml"
# train_dataset_cfg = OmegaConf.load(dataset_cfg)
# # datapipe = hydra.utils.instantiate(train_dataset_cfg)
# image_transform_cfg = OmegaConf.load("configs/processer/qwen_448_transform.yaml")
# image_transform = hydra.utils.instantiate(image_transform_cfg)
# def build_dataloader(dataset_cfg, image_transform, tokenizer, batch_size, dataloader_num_workers=4):
#     dataset = hydra.utils.instantiate(dataset_cfg, image_transform=image_transform, tokenizer=tokenizer)
#     # mp_service = MultiProcessingReadingService(num_workers=dataloader_num_workers)
#     # dist_service = DistributedReadingService()
#     # reading_service = SequentialReadingService(dist_service, mp_service)
#     dataloader = DataLoader2(dataset, reading_service=None)
#     # sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, shuffle=True)
#     # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=dataloader_num_workers, sampler=sampler)
#     return dataloader
# train_dataloader = build_dataloader(dataset_cfg=train_dataset_cfg,
#                                     image_transform=image_transform,
#                                     tokenizer=None,
#                                     batch_size=4,
#                                     dataloader_num_workers=1)
# # # 10 iters
# # for idx, data in enumerate(datapipe):
# #     if idx == 3:
# #         break
# #     print(data)
# #     # print(data[1].read())
# #     # print(len(data))
# #     # print(type(data[0]))
# #     # print(type(data[1]))
# #     # print(data[1].read())
# #     # print(type(data[2]))
# #     # image = Image.open(data[2].read()).convert('RGB')
# #     # print(image)
# #     print('---')

# # 3 iters
# dataiter = iter(train_dataloader)
# for i in range(3):
#     print(i)
#     data = next(dataiter)
#     print(data)
#     print(data['images'].shape)


# import torch
# import torch.nn as nn

# # 参数
# embed_dim = 4096
# num_heads = 8  # 确保可整除4096

# # 创建 MultiheadAttention 模块
# attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=False)

# # 模拟输入数据
# batch_size = 8
# seq_len_q = 256
# seq_len_kv = 1024

# # query, key, value
# q = torch.randn(256, 4096)  # 注意维度顺序
# x = torch.randn(1024, 8, 4096)
# s_pos_embed = torch.randn(256, 4096)
# pos_embed = torch.randn(1024, 4096)
# N = 8

# # 模拟位置编码和重复操作
# def _repeat(query, N: int):
#     return query.unsqueeze(1).repeat(1, N, 1)

# # 调整位置编码的形状并加到查询和键值对上
# q_pos = _repeat(q, 8) + pos_embed[:seq_len_q].unsqueeze(1)
# x_pos = x + pos_embed.unsqueeze(1)

# # 注意力掩码（如果有的话）
# attn_mask = None  # 你可以定义一个适当的掩码

# # 执行多头注意力
# output = attn(_repeat(q, N) + s_pos_embed.unsqueeze(1), x + pos_embed.unsqueeze(1), x, attn_mask=attn_mask)[0]

# print("Output shape:", output.shape)


# check F.pad is 
import torch
import torch.nn.functional as F

# 1. 1D padding
x = torch.randn(2, 3)
print(x)
# pad with 1
x_pad = F.pad(x, (1, 1), mode="constant", value=0)
print(x_pad)
# pad with 0
x_pad = F.pad(x, (1, 1), mode="constant", value=1)
print(x_pad)

# 2. 2D padding
x = torch.randn(2, 3, 4)
print(x)
# pad with 1
x_pad = F.pad(x, (1, 1, 1, 1), mode="constant", value=0)
print(x_pad)
# pad with 0
x_pad = F.pad(x, (1, 1, 1, 1), mode="constant", value=1)
print(x_pad)
