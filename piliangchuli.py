#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量任务管理器 (Batch Runner)
功能：扫描指定目录下的所有病人数据，并自动调用 run_full_pipeline.py 进行批量处理。
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# ================= 配置区域 (请修改这里) =================

# 1. 你的数据总仓库在哪里？
#    程序会扫描这个文件夹下的所有子文件夹（作为DICOM）或 .nii.gz 文件
INPUT_ROOT_DIR = r"C:\LocalProject\zhongshan_lung_CT\dicom" 

# 2. 结果存到哪里？
OUTPUT_ROOT_DIR = r"C:\LocalProject\zhongshan_lung_CT\lung_seg_api\batch_output"

# 3. 核心参数配置 (针对内窥镜优化)
CONFIG = {
    "script_path": "run_full_pipeline.py", # 核心脚本名
    "models": "airway",                    # 只跑气管模型
    "device": "cuda",                      # 使用显卡 (如果没有显卡请改为 "cpu")
    "margin": "10",                        # 裁剪边缘
    "extract_tree": True,                  # 是否提取树结构 (True/False)
    "crop_type": "lung"                    # 裁剪类型
}

# =======================================================

def get_python_exec():
    return sys.executable

def find_targets(root_dir):
    """扫描目录寻找待处理的病人数据"""
    root = Path(root_dir)
    if not root.exists():
        print(f"[错误] 找不到输入目录: {root}")
        return [], []

    # 1. 寻找DICOM目录 (假设每个子文件夹是一个病人)
    dicom_dirs = [d for d in root.iterdir() if d.is_dir()]
    
    # 2. 寻找NIfTI文件
    nifti_files = list(root.glob("*.nii.gz")) + list(root.glob("*.nii"))
    
    return dicom_dirs, nifti_files

def run_single_case(target_path, output_dir, total, current):
    """调用主流程处理单个案例"""
    cmd = [
        get_python_exec(),
        CONFIG["script_path"],
        "-i", str(target_path),
        "-o", str(output_dir),
        "--models", CONFIG["models"],
        "--device", CONFIG["device"],
        "--margin", CONFIG["margin"],
        "--crop-type", CONFIG["crop_type"]
    ]
    
    if CONFIG["extract_tree"]:
        cmd.append("--extract-airway-tree")

    print("\n" + "="*60)
    print(f">>> 正在处理进度 [{current}/{total}]: {target_path.name}")
    print("="*60)

    try:
        # 调用 run_full_pipeline.py
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"√ [成功] {target_path.name}")
            return True
        else:
            print(f"X [失败] {target_path.name}")
            return False
    except Exception as e:
        print(f"X [异常] {e}")
        return False

def main():
    print(">>> 启动批量内窥镜数据处理任务...")
    
    # 检查主脚本是否存在
    if not Path(CONFIG["script_path"]).exists():
        print(f"[错误] 找不到核心脚本: {CONFIG['script_path']}")
        print("请确保 run_batch_task.py 和 run_full_pipeline.py 在同一个文件夹里。")
        input("按回车键退出...")
        return

    # 扫描数据
    dicom_dirs, nifti_files = find_targets(INPUT_ROOT_DIR)
    all_targets = dicom_dirs + nifti_files
    total_count = len(all_targets)

    if total_count == 0:
        print(f"[警告] 在 {INPUT_ROOT_DIR} 下没有找到任何文件夹或 .nii 文件。")
        input("按回车键退出...")
        return

    print(f"共发现 {total_count} 个待处理案例。")
    print(f"- DICOM 目录: {len(dicom_dirs)} 个")
    print(f"- NIfTI 文件: {len(nifti_files)} 个")
    print("5秒后开始处理... (按 Ctrl+C 可终止)")
    time.sleep(5)

    success_count = 0
    
    # 开始循环
    for i, target in enumerate(all_targets, 1):
        if run_single_case(target, OUTPUT_ROOT_DIR, total_count, i):
            success_count += 1

    print("\n" + "="*60)
    print("批量任务完成汇报")
    print("="*60)
    print(f"总任务数: {total_count}")
    print(f"成功: {success_count}")
    print(f"失败: {total_count - success_count}")
    print(f"结果目录: {OUTPUT_ROOT_DIR}")
    print("="*60)
    input("按回车键关闭...")

if __name__ == "__main__":
    main()