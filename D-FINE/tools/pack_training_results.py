#!/usr/bin/env python3
"""训练后自动评估和打包脚本"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='训练输出目录，如 ./output/exp_s_obj2coco_d_bg4_no_A')
    parser.add_argument('--config', required=True, help='训练配置文件，如 ../config/volleyball_s_obj2coco_d_bg4.yml')
    parser.add_argument('--pack-dir', default='E:/数据集/model', help='打包输出目录')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"错误: 输出目录不存在 {output_dir}")
        sys.exit(1)

    # 1. 查找最佳模型
    best_model = output_dir / 'best_stg2.pth'
    if not best_model.exists():
        best_model = output_dir / 'best_stg1.pth'
    if not best_model.exists():
        print(f"错误: 未找到best模型文件")
        sys.exit(1)
    print(f"找到模型: {best_model}")

    # 2. 读取log.txt获取最佳epoch和AP
    log_file = output_dir / 'log.txt'
    best_ap, best_epoch = 0, -1
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    ap = data.get('test_coco_eval_bbox', [0])[0]
                    if ap > best_ap:
                        best_ap = ap
                        best_epoch = data['epoch']
    print(f"最佳结果: AP={best_ap:.4f} @ epoch {best_epoch}")

    # 3. 运行模型测试
    print("\n运行模型测试...")
    test_cmd = [
        'python', 'train.py',
        '-c', args.config,
        '--test-only',
        '-r', str(best_model),
    ]
    subprocess.run(test_cmd, check=True)

    # 4. 运行OA评估
    print("\n运行OA评估...")
    oa_script = Path('benchmark/scripts/eval_overactivation.py')
    if oa_script.exists():
        oa_cmd = [
            'python', str(oa_script),
            '--model-path', str(best_model),
            '--config', args.config,
            '--negative-dir', '../coco/images/negative_samples',
            '--output', str(output_dir / 'oa_report.json'),
        ]
        try:
            subprocess.run(oa_cmd, check=True)
        except:
            print("警告: OA评估失败，继续打包")

    # 5. 打包文件
    print("\n打包文件...")
    exp_name = output_dir.name
    pack_name = f"{exp_name}_epoch{best_epoch}_AP{best_ap:.4f}".replace('.', '_')
    pack_path = Path(args.pack_dir) / pack_name
    pack_path.mkdir(parents=True, exist_ok=True)

    # 复制文件
    files_to_pack = [
        ('best_stg2.pth', '模型文件'),
        ('best_stg1.pth', '模型文件'),
        ('best_oa.pth', '模型文件'),
        ('log.txt', '训练日志'),
        ('oa_report.json', 'OA报告'),
    ]

    for fname, desc in files_to_pack:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, pack_path / fname)
            print(f"  ✓ {desc}: {fname}")

    # 复制TensorBoard日志
    summary_dir = output_dir / 'summary'
    if summary_dir.exists():
        shutil.copytree(summary_dir, pack_path / 'summary', dirs_exist_ok=True)
        print(f"  ✓ TensorBoard日志: summary/")

    # 创建README
    readme = pack_path / 'README.txt'
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"""训练结果打包
实验名称: {exp_name}
最佳Epoch: {best_epoch}
最佳AP50:95: {best_ap:.4f}
配置文件: {args.config}
打包时间: {Path(__file__).stat().st_mtime}

文件说明:
- best_stg2.pth: 最佳模型（stg2阶段）
- best_stg1.pth: 最佳模型（stg1阶段）
- best_oa.pth: OA最优模型（如果有）
- log.txt: 完整训练日志
- oa_report.json: OA评估报告
- summary/: TensorBoard训练曲线
""")

    # 压缩
    print(f"\n压缩到: {pack_path}.zip")
    shutil.make_archive(str(pack_path), 'zip', pack_path.parent, pack_path.name)
    print(f"✓ 打包完成: {pack_path}.zip")

if __name__ == '__main__':
    main()
