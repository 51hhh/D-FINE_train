#!/usr/bin/env python3
'''
python tools/pack_training_results.py \
    --output-dir ./output/exp_s_obj2coco_d_bg4_no_A \
    --config ../config/volleyball_s_obj2coco_d_bg4.yml \
    --pack-dir /personal/
'''

"""训练后自动评估和打包脚本。直接运行 test-only，完成 AP 与 OA 测评后统一打包。"""
import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_oa_result_line(line):
    """解析 test-only 输出中的 OA 结果。"""
    line = line.strip()
    if not line or "Over-Activation @" not in line:
        return None

    prefix, metrics_str = line.split(":", 1)
    try:
        metrics = ast.literal_eval(metrics_str.strip())
    except (SyntaxError, ValueError):
        return None

    threshold_str = prefix.split("@", 1)[1].strip()
    phase = "standalone"
    epoch = -1
    if "[epoch " in threshold_str:
        threshold, rest = threshold_str.split("[epoch ", 1)
        epoch = int(rest.rstrip("] "))
        phase = "epoch"
    elif "[final]" in threshold_str:
        threshold = threshold_str.split("[final]", 1)[0]
        phase = "final"
    else:
        threshold = threshold_str

    return {
        "epoch": epoch,
        "phase": phase,
        "threshold": float(threshold.strip()),
        **metrics,
    }


def extract_oa_results(log_file):
    """从日志中提取 OA 结果，仅作兼容保留。"""
    results = []
    if not log_file.exists():
        return results
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            parsed = parse_oa_result_line(line)
            if parsed is not None:
                results.append(parsed)
    return results


def generate_oa_report(oa_results, output_dir):
    if not oa_results:
        return None

    best_oa = min(oa_results, key=lambda x: x.get('oa_fppi', float('inf')))
    report = {
        'total_evaluations': len(oa_results),
        'best_result': best_oa,
        'all_results': oa_results,
        'summary': {
            'best_epoch': best_oa['epoch'],
            'best_oa_fppi': best_oa.get('oa_fppi', 0),
            'best_oa_clean_rate': best_oa.get('oa_clean_rate', 0),
            'best_oa_max_score': best_oa.get('oa_max_score', 0),
        },
    }

    json_path = output_dir / 'oa_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_lines = [
        '# Over-Activation 评估报告\n',
        f'## 最佳结果 (Epoch {best_oa["epoch"]})\n',
        f'- **OA FPPI**: {best_oa.get("oa_fppi", 0):.4f}',
        f'- **OA Clean Rate**: {best_oa.get("oa_clean_rate", 0):.4f}',
        f'- **OA Max Score**: {best_oa.get("oa_max_score", 0):.4f}\n',
        f'## 评估历史 (共 {len(oa_results)} 次)\n',
        '| Epoch | Phase | OA FPPI | Clean Rate | Max Score |',
        '|-------|-------|---------|------------|-----------|',
    ]
    for result in oa_results:
        md_lines.append(
            f'| {result["epoch"]} | {result.get("phase", "standalone")} | {result.get("oa_fppi", 0):.4f} | '
            f'{result.get("oa_clean_rate", 0):.4f} | {result.get("oa_max_score", 0):.4f} |'
        )

    md_path = output_dir / 'oa_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    return report


def load_oa_eval_settings(config_path):
    settings = {
        'negative_img_dir': '../coco/images/negative_samples',
        'oa_conf_threshold': 0.3,
        'input_size': 640,
    }
    if not Path(config_path).exists():
        return settings

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    settings['negative_img_dir'] = cfg.get('negative_img_dir', settings['negative_img_dir'])
    settings['oa_conf_threshold'] = float(cfg.get('oa_conf_threshold', settings['oa_conf_threshold']))
    input_size = cfg.get('eval_spatial_size', settings['input_size'])
    if isinstance(input_size, (list, tuple)):
        input_size = input_size[0]
    settings['input_size'] = int(input_size)
    return settings


def build_test_only_command(model_path, config_path, oa_settings):
    return [
        'python', 'train.py',
        '-c', Path(config_path).as_posix(),
        '--test-only',
        '-r', Path(model_path).as_posix(),
        '-u',
        'eval_overactivation=True',
        f'negative_img_dir={oa_settings["negative_img_dir"]}',
        f'oa_conf_threshold={oa_settings["oa_conf_threshold"]}',
        f'eval_spatial_size=[{oa_settings["input_size"]},{oa_settings["input_size"]}]',
    ]


def extract_best_metrics_from_log(log_file):
    best_ap, best_epoch = 0.0, -1
    if not log_file.exists():
        return best_ap, best_epoch

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = data.get('test_coco_eval_bbox', [0])[0]
            if ap > best_ap:
                best_ap = ap
                best_epoch = data.get('epoch', -1)
    return best_ap, best_epoch


def run_test_and_oa(best_model, config_path, oa_settings, output_dir):
    cmd = build_test_only_command(best_model, config_path, oa_settings)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

    eval_log = output_dir / 'eval_stdout.txt'
    eval_log.write_text(result.stdout, encoding='utf-8')
    if result.stderr:
        (output_dir / 'eval_stderr.txt').write_text(result.stderr, encoding='utf-8')

    oa_results = []
    for line in result.stdout.splitlines():
        parsed = parse_oa_result_line(line)
        if parsed is not None:
            oa_results.append(parsed)

    oa_json_path = output_dir / 'oa_eval_result.json'
    if oa_results:
        with open(oa_json_path, 'w', encoding='utf-8') as f:
            json.dump(oa_results[-1], f, indent=2, ensure_ascii=False)

    return oa_results, result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='训练输出目录，如 ./output/exp_s_obj2coco_d_bg4_no_A')
    parser.add_argument('--config', required=True, help='训练配置文件，如 ../config/volleyball_s_obj2coco_d_bg4.yml')
    parser.add_argument('--pack-dir', default='E:/数据集/model', help='打包输出目录')
    parser.add_argument('--log-file', help='训练日志文件路径（可选，默认为 output-dir/log.txt）')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = Path(args.config)
    if not output_dir.exists():
        print(f"错误: 输出目录不存在 {output_dir}")
        sys.exit(1)

    best_model = output_dir / 'best_stg2.pth'
    if not best_model.exists():
        best_model = output_dir / 'best_stg1.pth'
    if not best_model.exists():
        print('错误: 未找到best模型文件')
        sys.exit(1)
    print(f'找到模型: {best_model}')

    log_file = Path(args.log_file) if args.log_file else (output_dir / 'log.txt')
    best_ap, best_epoch = extract_best_metrics_from_log(log_file)
    print(f'最佳结果: AP={best_ap:.4f} @ epoch {best_epoch}')

    oa_settings = load_oa_eval_settings(config_path)
    print('\n运行模型测试 + OA 测评...')
    oa_results, _ = run_test_and_oa(best_model, config_path, oa_settings, output_dir)
    if oa_results:
        generate_oa_report(oa_results, output_dir)
        best_oa = min(oa_results, key=lambda x: x.get('oa_fppi', float('inf')))
        print(f"  ✓ OA 评估完成: FPPI={best_oa.get('oa_fppi', 0):.4f}")
    else:
        print('  ⚠ test-only 输出中未找到 OA 结果')

    print('\n打包文件...')
    exp_name = output_dir.name
    pack_name = f"{exp_name}_epoch{best_epoch}_AP{best_ap:.4f}".replace('.', '_')
    pack_path = Path(args.pack_dir) / pack_name
    pack_path.mkdir(parents=True, exist_ok=True)

    files_to_pack = [
        ('best_stg2.pth', '模型文件'),
        ('best_stg1.pth', '模型文件'),
        ('best_oa.pth', '模型文件'),
        ('log.txt', '训练日志'),
        ('oa_report.json', 'OA评估报告(JSON)'),
        ('oa_report.md', 'OA评估报告(Markdown)'),
        ('oa_eval_result.json', 'OA评估结果'),
        ('eval_stdout.txt', 'test-only输出'),
        ('eval_stderr.txt', 'test-only错误输出'),
    ]

    for fname, desc in files_to_pack:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, pack_path / fname)
            print(f'  ✓ {desc}: {fname}')
        elif fname not in {'best_oa.pth', 'oa_report.json', 'oa_report.md', 'oa_eval_result.json', 'eval_stderr.txt'}:
            print(f'  ⚠ 未找到: {fname}')

    summary_dir = output_dir / 'summary'
    if summary_dir.exists():
        shutil.copytree(summary_dir, pack_path / 'summary', dirs_exist_ok=True)
        print('  ✓ TensorBoard日志: summary/')

    readme = pack_path / 'README.txt'
    oa_summary = ''
    if oa_results:
        best_oa = min(oa_results, key=lambda x: x.get('oa_fppi', float('inf')))
        oa_summary = f"""
OA评估结果:
- OA FPPI: {best_oa.get('oa_fppi', 0):.4f}
- OA Clean Rate: {best_oa.get('oa_clean_rate', 0):.4f}
- OA Max Score: {best_oa.get('oa_max_score', 0):.4f}
- OA 阈值: {best_oa.get('threshold', 0):.4f}
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
- oa_report.json / oa_report.md: OA评估报告
- oa_eval_result.json: 当前 best checkpoint 的 OA 结果
- eval_stdout.txt / eval_stderr.txt: test-only 原始输出
- summary/: TensorBoard训练曲线
""")

    print(f'\n压缩到: {pack_path}.zip')
    shutil.make_archive(str(pack_path), 'zip', pack_path.parent, pack_path.name)
    print(f'✓ 打包完成: {pack_path}.zip')


if __name__ == '__main__':
    main()
