#!/usr/bin/env python3
'''
python tools/pack_training_results.py \
    --output-dir ./output/exp_s_obj2coco_d_bg4_no_A \
    --config ../config/volleyball_s_obj2coco_d_bg4.yml \
    --pack-dir /personal/
'''

"""训练后自动评估和打包脚本"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
import shutil

def extract_oa_results(log_file):
    """从训练日志中提取 OA 评估结果"""
    oa_results = []
    pattern = r"Over-Activation @([\d.]+) \[epoch (\d+)\]: (\{.*?\})"

    if not log_file.exists():
        return oa_results

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                threshold = float(match.group(1))
                epoch = int(match.group(2))
                metrics_str = match.group(3).replace("'", '"')
                try:
                    metrics = json.loads(metrics_str)
                    oa_results.append({
                        'epoch': epoch,
                        'threshold': threshold,
                        **metrics
                    })
                except json.JSONDecodeError:
                    continue

    return oa_results

def generate_oa_report(oa_results, output_dir):
    """生成 OA 评估报告"""
    if not oa_results:
        return None

    # 找到最佳 OA 结果（最低 fppi）
    best_oa = min(oa_results, key=lambda x: x.get('oa_fppi', float('inf')))

    # 生成 JSON 报告
    report = {
        'total_evaluations': len(oa_results),
        'best_result': best_oa,
        'all_results': oa_results,
        'summary': {
            'best_epoch': best_oa['epoch'],
            'best_oa_fppi': best_oa.get('oa_fppi', 0),
            'best_oa_clean_rate': best_oa.get('oa_clean_rate', 0),
            'best_oa_max_score': best_oa.get('oa_max_score', 0),
        }
    }

    json_path = output_dir / 'oa_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 生成 Markdown 报告
    md_lines = [
        '# Over-Activation 评估报告\n',
        f'## 最佳结果 (Epoch {best_oa["epoch"]})\n',
        f'- **OA FPPI**: {best_oa.get("oa_fppi", 0):.4f}',
        f'- **OA Clean Rate**: {best_oa.get("oa_clean_rate", 0):.4f}',
        f'- **OA Max Score**: {best_oa.get("oa_max_score", 0):.4f}\n',
        f'## 评估历史 (共 {len(oa_results)} 次)\n',
        '| Epoch | OA FPPI | Clean Rate | Max Score |',
        '|-------|---------|------------|-----------|',
    ]

    for result in sorted(oa_results, key=lambda x: x['epoch']):
        md_lines.append(
            f'| {result["epoch"]} | {result.get("oa_fppi", 0):.4f} | '
            f'{result.get("oa_clean_rate", 0):.4f} | {result.get("oa_max_score", 0):.4f} |'
        )

    md_path = output_dir / 'oa_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='训练输出目录，如 ./output/exp_s_obj2coco_d_bg4_no_A')
    parser.add_argument('--config', required=True, help='训练配置文件，如 ../config/volleyball_s_obj2coco_d_bg4.yml')
    parser.add_argument('--pack-dir', default='E:/数据集/model', help='打包输出目录')
    parser.add_argument('--log-file', help='训练日志文件路径（可选，默认为 output-dir/log.txt）')
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
    log_file = Path(args.log_file) if args.log_file else (output_dir / 'log.txt')
    best_ap, best_epoch = 0, -1
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and line.strip().startswith('{'):
                    try:
                        data = json.loads(line)
                        ap = data.get('test_coco_eval_bbox', [0])[0]
                        if ap > best_ap:
                            best_ap = ap
                            best_epoch = data['epoch']
                    except json.JSONDecodeError:
                        continue
    print(f"最佳结果: AP={best_ap:.4f} @ epoch {best_epoch}")

    # 2.5 提取 OA 评估结果
    print("\n提取 OA 评估结果...")
    oa_results = extract_oa_results(log_file)
    if oa_results:
        oa_report = generate_oa_report(oa_results, output_dir)
        print(f"  ✓ 找到 {len(oa_results)} 次 OA 评估")
        if oa_report:
            best_oa = oa_report['best_result']
            print(f"  ✓ 最佳 OA: FPPI={best_oa.get('oa_fppi', 0):.4f} @ epoch {best_oa['epoch']}")
    else:
        print("  ⚠ 未找到 OA 评估结果")

    # 3. 运行模型测试
    print("\n运行模型测试...")
    test_cmd = [
        'python', 'train.py',
        '-c', args.config,
        '--test-only',
        '-r', str(best_model),
    ]
    subprocess.run(test_cmd, check=True)

    # 4. 如果日志中没有 OA 数据，运行独立 OA 评估
    if not oa_results:
        print("\n日志中未找到 OA 数据，运行独立 OA 评估...")
        oa_script = Path('tools/eval_overactivation.py')
        if not oa_script.exists():
            oa_script = Path('benchmark/scripts/eval_overactivation.py')

        if oa_script.exists():
            oa_output = output_dir / 'oa_eval_result.json'
            oa_cmd = [
                'python', str(oa_script),
                '--model-path', str(best_model),
                '--config', args.config,
                '--negative-dir', '../coco/images/negative_samples',
                '--output', str(oa_output),
            ]
            try:
                subprocess.run(oa_cmd, check=True)
                # 读取评估结果并转换为统一格式
                if oa_output.exists():
                    with open(oa_output, 'r', encoding='utf-8') as f:
                        eval_result = json.load(f)
                    # 转换为 oa_results 格式
                    oa_results = [{
                        'epoch': best_epoch,
                        'threshold': 0.3,
                        'oa_fppi': eval_result.get('oa_fppi', 0),
                        'oa_clean_rate': eval_result.get('oa_clean_rate', 0),
                        'oa_max_score': eval_result.get('oa_max_score', 0),
                    }]
                    oa_report = generate_oa_report(oa_results, output_dir)
                    print(f"  ✓ OA 评估完成: FPPI={oa_results[0]['oa_fppi']:.4f}")
            except Exception as e:
                print(f"  ⚠ OA 评估失败: {e}")
        else:
            print(f"  ⚠ 未找到 OA 评估脚本: {oa_script}")
    else:
        print(f"  ✓ 使用日志中的 OA 数据（{len(oa_results)} 次评估）")

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
        ('oa_report.json', 'OA评估报告(JSON)'),
        ('oa_report.md', 'OA评估报告(Markdown)'),
    ]

    for fname, desc in files_to_pack:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, pack_path / fname)
            print(f"  ✓ {desc}: {fname}")
        elif 'oa_report' not in fname:  # OA 报告可能不存在
            print(f"  ⚠ 未找到: {fname}")

    # 复制TensorBoard日志
    summary_dir = output_dir / 'summary'
    if summary_dir.exists():
        shutil.copytree(summary_dir, pack_path / 'summary', dirs_exist_ok=True)
        print(f"  ✓ TensorBoard日志: summary/")

    # 创建README
    readme = pack_path / 'README.txt'
    oa_summary = ""
    if oa_results:
        best_oa = min(oa_results, key=lambda x: x.get('oa_fppi', float('inf')))
        oa_summary = f"""
OA评估结果:
- 最佳OA Epoch: {best_oa['epoch']}
- OA FPPI: {best_oa.get('oa_fppi', 0):.4f}
- OA Clean Rate: {best_oa.get('oa_clean_rate', 0):.4f}
- OA Max Score: {best_oa.get('oa_max_score', 0):.4f}
- 总评估次数: {len(oa_results)}
"""

    with open(readme, 'w', encoding='utf-8') as f:
        f.write(f"""训练结果打包
实验名称: {exp_name}
最佳Epoch: {best_epoch}
最佳AP50:95: {best_ap:.4f}
配置文件: {args.config}
{oa_summary}
文件说明:
- best_stg2.pth: 最佳模型（stg2阶段）
- best_stg1.pth: 最佳模型（stg1阶段）
- best_oa.pth: OA最优模型（如果有）
- log.txt: 完整训练日志
- oa_report.json: OA评估报告（JSON格式）
- oa_report.md: OA评估报告（Markdown格式）
- summary/: TensorBoard训练曲线
""")

    # 压缩
    print(f"\n压缩到: {pack_path}.zip")
    shutil.make_archive(str(pack_path), 'zip', pack_path.parent, pack_path.name)
    print(f"✓ 打包完成: {pack_path}.zip")

if __name__ == '__main__':
    main()
