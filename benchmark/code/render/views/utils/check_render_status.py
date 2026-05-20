#!/usr/bin/env python3
"""
检查渲染状态和统计
生成JSON报告
"""
import os
import json
from datetime import datetime
from pathlib import Path


def check_render_status(output_root):
    """检查渲染状态并生成统计"""
    
    datasets = {
        'physxanything_mobility': os.path.join(output_root, 'physxanything_mobility'),
        'physxanything_verse': os.path.join(output_root, 'physxanything_verse'),
    }
    
    mesh_roots = {
        'physxanything_mobility': 'physx_result/physxanything_mobility',
        'physxanything_verse': 'physx_result/physxanything_verse',
    }
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': {},
        'summary': {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'pending': 0,
        }
    }
    
    for dataset_name, output_dir in datasets.items():
        mesh_root = mesh_roots[dataset_name]
        
        if not os.path.exists(mesh_root):
            continue
        
        dataset_result = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'pending': 0,
            'completed_items': [],
            'failed_items': [],
            'pending_items': [],
        }
        
        # 遍历所有mesh
        for item in sorted(os.listdir(mesh_root)):
            mesh_path = os.path.join(mesh_root, item, 'mesh.glb')
            if not os.path.isfile(mesh_path):
                continue
            
            dataset_result['total'] += 1
            
            # 检查输出
            save_dir = os.path.join(output_dir, item)
            transforms_json = os.path.join(save_dir, 'transforms.json')
            
            if os.path.exists(transforms_json):
                # 检查是否真的完成（25张图片）
                png_files = list(Path(save_dir).glob('*.png'))
                if len(png_files) >= 25:
                    dataset_result['completed'] += 1
                    dataset_result['completed_items'].append(item)
                else:
                    dataset_result['failed'] += 1
                    dataset_result['failed_items'].append({
                        'name': item,
                        'reason': f'Incomplete: only {len(png_files)} images'
                    })
            else:
                # 检查是否有部分输出（可能正在渲染或失败）
                if os.path.exists(save_dir):
                    png_files = list(Path(save_dir).glob('*.png'))
                    if len(png_files) > 0:
                        # 有部分输出，可能正在渲染
                        dataset_result['pending'] += 1
                        dataset_result['pending_items'].append({
                            'name': item,
                            'status': f'Rendering: {len(png_files)}/25 images'
                        })
                    else:
                        # 文件夹存在但没有输出
                        dataset_result['failed'] += 1
                        dataset_result['failed_items'].append({
                            'name': item,
                            'reason': 'No output files'
                        })
                else:
                    # 还没开始
                    dataset_result['pending'] += 1
                    dataset_result['pending_items'].append({
                        'name': item,
                        'status': 'Not started'
                    })
        
        results['datasets'][dataset_name] = dataset_result
        
        # 更新总计
        results['summary']['total'] += dataset_result['total']
        results['summary']['completed'] += dataset_result['completed']
        results['summary']['failed'] += dataset_result['failed']
        results['summary']['pending'] += dataset_result['pending']
    
    return results


def main():
    output_root = 'benchmark/benchmark_assets/rendered_views/description'
    
    print('检查渲染状态...')
    results = check_render_status(output_root)
    
    # 保存JSON
    json_file = os.path.join(output_root, 'render_status.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"渲染状态报告 - {results['timestamp']}")
    print(f"{'='*60}")
    print(f"\n总体统计：")
    print(f"  总计：{results['summary']['total']}")
    print(f"  已完成：{results['summary']['completed']} ({results['summary']['completed']/results['summary']['total']*100:.1f}%)")
    print(f"  进行中：{results['summary']['pending']} ({results['summary']['pending']/results['summary']['total']*100:.1f}%)")
    print(f"  失败：{results['summary']['failed']} ({results['summary']['failed']/results['summary']['total']*100:.1f}%)")
    
    print(f"\n分数据集统计：")
    for dataset_name, stats in results['datasets'].items():
        print(f"\n  {dataset_name}:")
        print(f"    总计：{stats['total']}")
        print(f"    已完成：{stats['completed']}")
        print(f"    进行中：{stats['pending']}")
        print(f"    失败：{stats['failed']}")
    
    if results['summary']['failed'] > 0:
        print(f"\n失败详情：")
        for dataset_name, stats in results['datasets'].items():
            if stats['failed'] > 0:
                print(f"\n  {dataset_name}:")
                for item in stats['failed_items'][:10]:  # 只显示前10个
                    if isinstance(item, dict):
                        print(f"    - {item['name']}: {item['reason']}")
                    else:
                        print(f"    - {item}")
                if len(stats['failed_items']) > 10:
                    print(f"    ... 还有 {len(stats['failed_items']) - 10} 个")
    
    print(f"\n详细报告已保存至：{json_file}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
