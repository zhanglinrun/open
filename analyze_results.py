"""
实验结果分析脚本
对比不同配置下的性能
"""

import torch
import numpy as np
from typing import Dict, List


def load_results(checkpoint_path: str) -> Dict:
    """加载实验结果"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return {
        'path': checkpoint_path,
        'epoch': checkpoint.get('epoch', -1),
        'open_f1': checkpoint.get('open_f1', -1),
        'known_acc': checkpoint.get('known_acc', -1),
        'open_precision': checkpoint.get('open_precision', -1),
        'open_recall': checkpoint.get('open_recall', -1),
        'open_acc': checkpoint.get('open_acc', -1),
    }


def print_comparison(results: List[Dict]):
    """打印对比结果"""
    print("=" * 100)
    print("开集识别实验结果对比")
    print("=" * 100)
    print(f"{'实验配置':<20} {'open_f1':<12} {'Precision':<12} {'Recall':<12} {'Known Acc':<12} {'Open Acc':<12}")
    print("-" * 100)

    for result in results:
        print(f"{result['path']:<20} "
              f"{result['open_f1']:.4f}     "
              f"{result['open_precision']:.4f}     "
              f"{result['open_recall']:.4f}     "
              f"{result['known_acc']:.4f}     "
              f"{result['open_acc']:.4f}")

    print("=" * 100)


def main():
    """主函数"""
    # 定义实验配置
    experiments = [
        ('exp1_baseline.pt', 'Baseline (简单原型)'),
        ('exp2_rectified_proto.pt', '+Rectified Proto'),
        ('exp3_pseudo_unknown.pt', '+Pseudo Unknown'),
        ('exp4_boundary_loss.pt', '+Boundary Loss'),
        ('exp5_full_model.pt', 'Full Model'),
    ]

    # 加载结果
    results = []
    for path, desc in experiments:
        print(f"加载: {path}")
        try:
            result = load_results(path)
            result['desc'] = desc
            results.append(result)
            print(f"  open_f1={result['open_f1']:.4f}, "
                  f"known_acc={result['known_acc']:.4f}")
        except FileNotFoundError:
            print(f"  警告: 文件不存在,跳过")
        except Exception as e:
            print(f"  错误: {e}")

    # 打印对比
    if results:
        print_comparison(results)

        # 找出最佳配置
        best_f1 = max(results, key=lambda x: x['open_f1'])
        print(f"\n最佳配置 (按open_f1): {best_f1['desc']}")
        print(f"  open_f1: {best_f1['open_f1']:.4f}")
        print(f"  known_acc: {best_f1['known_acc']:.4f}")

        # 分析改进效果
        baseline = next((r for r in results if 'baseline' in r['path']), None)
        if baseline:
            print(f"\n改进分析 (相对于Baseline):")
            for result in results:
                if result != baseline:
                    improvement = (result['open_f1'] - baseline['open_f1']) / max(baseline['open_f1'], 1e-6) * 100
                    print(f"  {result['desc']:<30} : {improvement:+.1f}%")
    else:
        print("\n没有找到可用的结果文件!")
        print("请先运行 run_experiments.sh")


if __name__ == "__main__":
    main()
