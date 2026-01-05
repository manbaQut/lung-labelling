#!/usr/bin/env python
"""
完整Pipeline示例脚本
演示从预处理到后处理的完整流程
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.logger import get_logger

logger = get_logger("pipeline")


def run_command(cmd, description):
    """运行命令并记录日志"""
    logger.info("="*60)
    logger.info(description)
    logger.info("="*60)
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        logger.error(f"命令执行失败: {description}")
        return False
    return True


def run_pipeline(input_file, model_types, min_volume=150, device='cuda', margin=5, crop_type='lung', extract_airway_tree=False, output_dir=None, filename_prefix=None):
    """
    运行完整pipeline
    
    Args:
        input_file: 输入文件路径 (DICOM目录或NIfTI文件)
        model_types: 模型类型列表 ['lung_lesion', 'airway', 'vessel']
        min_volume: 最小病灶体积阈值
        device: 推理设备 (cuda/cpu)
        margin: 裁剪边界扩展像素数
        crop_type: 裁剪类型 (lung/rib)
        extract_airway_tree: 是否提取气管树结构 (仅当包含airway模型时有效)
        output_dir: 输出目录路径 (默认: 输入路径的父目录)
        filename_prefix: 输出文件名前缀 (默认: 从输入路径推导)
    """
    input_path = Path(input_file).absolute()
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        return False
    
    # 检测输入类型：DICOM目录或NIfTI文件
    is_dicom = input_path.is_dir()
    is_nifti = input_path.is_file() and str(input_path).lower().endswith(('.nii', '.nii.gz'))
    
    if not is_dicom and not is_nifti:
        logger.error(f"输入必须是DICOM目录或NIfTI文件: {input_path}")
        return False
    
    # 确定case_name和workspace
    if is_dicom:
        default_case_name = input_path.name
        default_workspace = input_path.parent
        input_type = "DICOM目录"
    else:
        # 正确处理NIfTI文件名（支持.nii和.nii.gz）
        if str(input_path).lower().endswith('.nii.gz'):
            default_case_name = input_path.name.replace('.nii.gz', '').replace('_0000', '')
        else:
            default_case_name = input_path.name.replace('.nii', '').replace('_0000', '')
        default_workspace = input_path.parent.parent
        input_type = "NIfTI文件"
    
    # 使用用户指定的输出目录和文件名前缀，或使用默认值
    workspace = Path(output_dir).absolute() if output_dir else default_workspace
    case_name = filename_prefix if filename_prefix else default_case_name
    
    # 目录设置
    temp_nii_dir = workspace / "temp_nii"
    preprocess_dir = workspace / "preprocessed"
    predictions_dir = workspace / "predictions"
    final_dir = workspace / "final"
    
    temp_nii_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info(f"开始处理: {case_name}")
    logger.info("="*60)
    logger.info(f"输入类型: {input_type}")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {workspace}")
    logger.info(f"文件前缀: {case_name}")
    logger.info(f"模型类型: {', '.join(model_types)}")
    logger.info(f"裁剪类型: {crop_type}, margin: {margin}")
    logger.info(f"推理设备: {device}")
    logger.info(f"最小体积: {min_volume} mm³")
    logger.info("="*60)
    
    # 步骤0: DICOM转NIfTI (如果需要)
    if is_dicom:
        nifti_file = temp_nii_dir / f"{case_name}_0000.nii.gz"
        
        logger.info("="*60)
        logger.info("[0] DICOM → NIfTI转换")
        logger.info("="*60)
        logger.info(f"DICOM目录: {input_path}")
        logger.info(f"输出NIfTI: {nifti_file}")
        
        # 使用inference.py中的DICOM转换函数
        try:
            from src.inference import load_dicom, save_nifti
            
            logger.info("正在转换DICOM...")
            img, origin, spacing = load_dicom(str(input_path))
            save_nifti(img, origin, spacing, str(nifti_file))
            logger.info(f"转换完成: {nifti_file}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"DICOM转换失败: {e}")
            return False
        
        input_for_preprocess = nifti_file
    else:
        input_for_preprocess = input_path
    
    # 步骤1: 预处理 - 裁剪到肺部区域
    preprocessed_file = preprocess_dir / f"{case_name}_0000.nii.gz"
    crop_indices_file = preprocess_dir / f"{case_name}_crop_indices.json"
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "preprocess_seg.py"),
        "-i", str(input_for_preprocess),
        "-o", str(preprocess_dir),  # 输出目录而不是文件路径
        "--margin", str(margin),
        "--crop-type", crop_type
    ]
    
    step_num = 1 if not is_dicom else 2
    
    # 计算总步骤数：预处理(1) + 推理(N) + 恢复(N) + 病灶拆分(0/1) + 气管树提取(0/1)
    extra_steps = 0
    if 'lung_lesion' in model_types:
        extra_steps += 1
    if 'airway' in model_types and extract_airway_tree:
        extra_steps += 1
    
    total_steps = (1 if not is_dicom else 2) + len(model_types) * 2 + extra_steps
    
    if not run_command(cmd, f"[{step_num}/{total_steps}] 预处理 - 裁剪到{crop_type}区域"):
        return False
    
    # 步骤2/3: 推理 - 运行分割模型
    prediction_files = {}
    for i, model_type in enumerate(model_types, start=1):
        pred_dir = predictions_dir / model_type
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "lung_seg.py"),
            "-i", str(preprocessed_file),
            "-o", str(pred_dir),
            "--model-type", model_type,
            "--device", device,
            "--no-postprocess"  # 在裁剪区域不做后处理
        ]
        
        current_step = step_num + i
        if not run_command(cmd, f"[{current_step}/{total_steps}] 推理 - {model_type}模型"):
            return False
        
        prediction_files[model_type] = pred_dir / f"{case_name}.nii.gz"
    
    # 步骤3/4/5: 后处理 - 恢复原始尺寸
    restored_files = {}
    step_offset = step_num + len(model_types)
    
    for i, model_type in enumerate(model_types, start=1):
        final_model_dir = final_dir / model_type
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        restored_file = final_model_dir / f"{case_name}.nii.gz"
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "postprocess_seg.py"),
            "--restore",
            "-i", str(prediction_files[model_type]),
            "--crop-indices", str(crop_indices_file),
            "-o", str(restored_file)
        ]
        
        current_step = step_offset + i
        if not run_command(cmd, f"[{current_step}/{total_steps}] 恢复 - {model_type}结果"):
            return False
        
        restored_files[model_type] = restored_file
    
    # 后处理步骤计数器
    current_final_step = step_offset + len(model_types) + 1
    
    # 病灶拆分（仅用于lung_lesion）
    if 'lung_lesion' in model_types:
        split_dir = final_dir / "lesion_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "postprocess_seg.py"),
            "-i", str(restored_files['lung_lesion']),
            "-o", str(split_dir),
            "--min-volume", str(min_volume)
        ]
        
        step_label = f"[{current_final_step}/{total_steps}] 病灶拆分"
        if not run_command(cmd, step_label):
            return False
        current_final_step += 1
    
    # 气管树结构提取（仅用于airway）
    airway_tree_json = None
    if 'airway' in model_types and extract_airway_tree:
        tree_dir = final_dir / "airway_tree"
        tree_dir.mkdir(parents=True, exist_ok=True)
        
        airway_tree_json = tree_dir / f"{case_name}_tree.json"
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "postprocess_seg.py"),
            "--extract-tree",
            "-i", str(restored_files['airway']),
            "-o", str(airway_tree_json)
        ]
        
        step_label = f"[{current_final_step}/{total_steps}] 气管树提取"
        if not run_command(cmd, step_label):
            logger.warning("气管树提取失败，继续...")
        else:
            logger.info(f"气管树JSON已保存: {airway_tree_json}")
    
    logger.info("="*60)
    logger.info("Pipeline完成!")
    logger.info("="*60)
    logger.info("输出文件:")
    for model_type, file_path in restored_files.items():
        logger.info(f"  {model_type}: {file_path}")
    
    if 'lung_lesion' in model_types:
        lesion_dir = final_dir / "lesion_split" / case_name / "lesions"
        if lesion_dir.exists():
            lesion_files = list(lesion_dir.glob("*.nii.gz"))
            logger.info(f"  病灶拆分: {len(lesion_files)} 个病灶文件")
            logger.info(f"    目录: {lesion_dir}")
    
    if 'airway' in model_types and extract_airway_tree and airway_tree_json:
        if airway_tree_json.exists():
            logger.info(f"  气管树JSON: {airway_tree_json}")
    
    logger.info("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='完整Pipeline: 预处理 → 推理 → 后处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # DICOM输入 - 运行完整pipeline（3个模型）
  python scripts/run_full_pipeline.py -i data/dicom/patient001
  
  # NIfTI输入 - 运行完整pipeline
  python scripts/run_full_pipeline.py -i data/nii/case001_0000.nii.gz
  
  # 仅运行病灶分割
  python scripts/run_full_pipeline.py -i data/dicom/patient001 --models lung_lesion
  
  # 运行病灶+气管分割
  python scripts/run_full_pipeline.py -i data/nii/case001_0000.nii.gz --models lung_lesion airway
  
  # 运行气管分割并提取树结构
  python scripts/run_full_pipeline.py -i data/dicom/patient001 --models airway --extract-airway-tree
  
  # 使用肋骨裁剪，更大的ROI
  python scripts/run_full_pipeline.py -i data/dicom/patient001 --crop-type rib --margin 10
  
  # 指定输出目录和文件名前缀
  python scripts/run_full_pipeline.py -i data/dicom/patient001 -o output/results --filename case001
  
  # 完整参数示例
  python scripts/run_full_pipeline.py -i data/dicom/patient001 -o output/custom --filename patient_A --models lung_lesion airway --extract-airway-tree
        """
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='输入DICOM目录或NIfTI文件'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['lung_lesion', 'airway', 'vessel'],
        default=['lung_lesion', 'airway', 'vessel'],
        help='要运行的模型 (默认: 全部3个)'
    )
    parser.add_argument(
        '--min-volume',
        type=float,
        default=0,
        help='最小病灶体积阈值 (mm³) (默认: 0)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='推理设备 (默认: cuda)'
    )
    parser.add_argument(
        '--margin',
        type=int,
        default=5,
        help='裁剪边界扩展像素数 (默认: 5)'
    )
    parser.add_argument(
        '--crop-type',
        choices=['lung', 'rib'],
        default='rib',
        help='裁剪类型: lung=肺部区域, rib=肋骨区域 (默认: rib)'
    )
    parser.add_argument(
        '--extract-airway-tree',
        action='store_true',
        help='提取气管树结构（仅当包含airway模型时有效）'
    )
    parser.add_argument(
        '--output', '-o',
        help='输出目录路径 (默认: 输入路径的父目录)'
    )
    parser.add_argument(
        '--filename',
        help='输出文件名前缀 (默认: 从输入路径推导，DICOM为目录名，NIfTI为文件名)'
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        input_file=args.input,
        model_types=args.models,
        min_volume=args.min_volume,
        device=args.device,
        margin=args.margin,
        crop_type=args.crop_type,
        extract_airway_tree=args.extract_airway_tree,
        output_dir=args.output,
        filename_prefix=args.filename
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

"""

python scripts/run_full_pipeline.py \
    -i C:\LocalProject\zhongshan_lung_CT\dicom\dicom \
    -o C:\LocalProject\zhongshan_lung_CT\lung_seg_api\output \
    --filename dicom_case_001 \
    --extract-airway-tree

"""