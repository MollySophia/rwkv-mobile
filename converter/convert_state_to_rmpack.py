#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import struct
import os
import json
from typing import Any, Dict, List, Tuple
from pathlib import Path
import argparse
import torch

class RWKVModelPacker:
    MAGIC_HEADER = b"RWKVMBLE"
    ALIGNMENT = 4096  # 4KB对齐

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.files: List[Tuple[str, int, int]] = []  # (filename, size, offset)
        self.binary_data: List[bytes] = []

    def add_config(self, key: str, value: Any):
        self.config[key] = value

    def add_file(self, file_path: str, file_name: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, 'rb') as f:
            data = f.read()
            size = len(data)
            self.binary_data.append(data)
            self.files.append((file_name, size, 0))  # offset稍后计算

    def add_file_from_bytes(self, data: bytes, file_name: str):
        size = len(data)
        self.binary_data.append(data)
        self.files.append((file_name, size, 0))  # offset稍后计算

    def _calculate_offsets(self):
        """计算所有文件的偏移量"""
        # 文件头大小
        offset = len(self.MAGIC_HEADER)

        # 配置项大小
        config_json = json.dumps(self.config, ensure_ascii=False).encode('utf-8')
        offset += 4 + len(config_json)  # 4字节存储配置项长度

        # 文件信息大小
        offset += 4  # 文件数量
        for filename, size, _ in self.files:
            offset += 4 + len(filename.encode('utf-8')) + 8  # 文件名长度 + 文件名 + size(8字节) + offset(8字节)

        # 对齐到4KB边界
        padding_size = (self.ALIGNMENT - (offset % self.ALIGNMENT)) % self.ALIGNMENT
        offset += padding_size

        # 更新文件偏移量,每个文件都4KB对齐
        for i, (filename, size, _) in enumerate(self.files):
            # 确保每个文件的offset都是4KB对齐的
            padding_size = (self.ALIGNMENT - (offset % self.ALIGNMENT)) % self.ALIGNMENT
            offset += padding_size
            self.files[i] = (filename, size, offset)
            offset += size

    def pack(self, output_path: str):
        """打包所有文件到输出路径"""
        self._calculate_offsets()

        with open(output_path, 'wb') as f:
            # 写入文件头
            f.write(self.MAGIC_HEADER)

            # 写入配置项
            config_json = json.dumps(self.config, ensure_ascii=False).encode('utf-8')
            f.write(struct.pack('<I', len(config_json)))  # 配置项长度
            f.write(config_json)

            # 写入文件信息
            f.write(struct.pack('<I', len(self.files)))  # 文件数量
            for filename, size, offset in self.files:
                filename_bytes = filename.encode('utf-8')
                f.write(struct.pack('<I', len(filename_bytes)))  # 文件名长度
                f.write(filename_bytes)  # 文件名
                f.write(struct.pack('<Q', size))  # 文件大小 (8字节)
                f.write(struct.pack('<Q', offset))  # 文件偏移量 (8字节)

            # 写入padding以对齐到4KB边界
            current_pos = f.tell()
            padding_size = (self.ALIGNMENT - (current_pos % self.ALIGNMENT)) % self.ALIGNMENT
            f.write(b'\0' * padding_size)

            # 写入二进制文件内容,每个文件都4KB对齐
            for i, data in enumerate(self.binary_data):
                # 确保当前位置是4KB对齐的
                current_pos = f.tell()
                padding_size = (self.ALIGNMENT - (current_pos % self.ALIGNMENT)) % self.ALIGNMENT
                f.write(b'\0' * padding_size)
                # 写入文件内容
                f.write(data)

        print(f"rmpack文件已打包到: {output_path}")
        print(f"总大小: {os.path.getsize(output_path)} 字节")
        print(f"包含文件数: {len(self.files)}")
        print(f"配置项: {self.config}")
        print(f"文件信息: {self.files}")

# [文件头] [配置项长度] [配置项JSON] [文件数量] [文件信息...] [padding] [二进制内容...]
#  8字节     4字节      N字节        4字节      M字节         P字节    ...

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--flower-template', action='store_true', help='enable User✿...✿\\nBot✿ chat template for this rmpack')
    args = parser.parse_args()

    packer = RWKVModelPacker()
    state = torch.load(args.input, map_location='cpu')
    num_heads, head_size, _ = state["blocks.0.att.time_state"].shape
    packer.add_config("hidden_size", num_heads * head_size)
    if args.flower_template:
        packer.add_config("flower_template", True)
    for k, v in state.items():
        assert "time_state" in k, "unsupported key in state pth file: " + k
        bytes_data = v.half().numpy().tobytes()
        packer.add_file_from_bytes(bytes_data, k)

    packer.pack(args.output)


if __name__ == "__main__":
    main()
