"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2025/11/15-15:07
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""


import os

import os
import shutil
from pathlib import Path
from typing import List, Set, Dict


class FileSynchronizer:
    def __init__(self, source_dir: str, target_dir: str):
        """
        文件同步器 - 将源目录中存在的文件复制到目标目录

        Args:
            source_dir: 源目录路径 (./images1)
            target_dir: 目标目录路径 (./images2)
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

        # 验证目录存在性
        if not self.source_dir.exists():
            raise ValueError(f"源目录不存在: {source_dir}")
        if not self.target_dir.exists():
            print(f"目标目录不存在，创建: {target_dir}")
            self.target_dir.mkdir(parents=True, exist_ok=True)

        print(f"文件同步器初始化完成")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        print("=" * 50)

    def get_missing_files(self, file_extensions: List[str] = None) -> Dict[str, List[Path]]:
        """
        获取目标目录中缺失的文件

        Args:
            file_extensions: 要检查的文件扩展名列表，None表示所有文件

        Returns:
            Dict: 缺失文件统计
        """
        if file_extensions is None:
            file_extensions = ['.txt']  # 默认只检查.txt文件

        # 获取源目录和目标目录中的文件
        source_files = self._get_files_with_extensions(self.source_dir, file_extensions)
        target_files = self._get_files_with_extensions(self.target_dir, file_extensions)

        # 提取文件名（不含路径）
        source_filenames = {f.name for f in source_files}
        target_filenames = {f.name for f in target_files}

        # 找出缺失的文件
        missing_filenames = source_filenames - target_filenames
        missing_files = [f for f in source_files if f.name in missing_filenames]

        return {
            'source_files': source_files,
            'target_files': target_files,
            'missing_files': missing_files,
            'source_count': len(source_files),
            'target_count': len(target_files),
            'missing_count': len(missing_files)
        }

    def _get_files_with_extensions(self, directory: Path, extensions: List[str]) -> List[Path]:
        """获取指定扩展名的文件列表"""
        files = []
        for ext in extensions:
            # 支持大小写变体
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(list(set(files)))

    def sync_files(self, file_extensions: List[str] = None,
                   dry_run: bool = False) -> Dict:
        """
        同步文件（复制缺失的文件）

        Args:
            file_extensions: 文件扩展名列表
            dry_run: 试运行模式（只显示将要执行的操作，不实际复制）

        Returns:
            Dict: 同步操作结果统计
        """
        # 分析缺失文件
        analysis = self.get_missing_files(file_extensions)

        if analysis['missing_count'] == 0:
            print("目标目录已包含所有源目录文件，无需同步")
            return analysis

        print(f"   同步分析结果:")
        print(f"   源目录文件数: {analysis['source_count']}")
        print(f"   目标目录文件数: {analysis['target_count']}")
        print(f"   缺失文件数: {analysis['missing_count']}")
        print()

        # 执行同步
        results = {
            'total_processed': 0,
            'successful_copies': 0,
            'failed_copies': 0,
            'copied_files': [],
            'failed_files': []
        }

        for missing_file in analysis['missing_files']:
            source_path = missing_file
            target_path = self.target_dir / missing_file.name

            if dry_run:
                print(f"[试运行] 将复制: {missing_file.name}")
                results['copied_files'].append(str(missing_file))
                continue

            try:
                # 执行文件复制
                shutil.copy2(source_path, target_path)
                results['successful_copies'] += 1
                results['copied_files'].append(str(missing_file))
                print(f"成功复制: {missing_file.name}")

            except Exception as e:
                results['failed_copies'] += 1
                results['failed_files'].append({
                    'file': str(missing_file),
                    'error': str(e)
                })
                print(f"复制失败: {missing_file.name} - {e}")

            results['total_processed'] += 1

        # 生成报告
        # if not dry_run:
        #     self._generate_sync_report(results, analysis)

        return {**analysis, **results}

    def _generate_sync_report(self, results: Dict, analysis: Dict):
        """生成同步报告"""
        report_path = self.target_dir / "file_sync_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("文件同步报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"同步时间: {__import__('datetime').datetime.now()}\n")
            f.write(f"源目录: {self.source_dir}\n")
            f.write(f"目标目录: {self.target_dir}\n\n")

            f.write("同步统计:\n")
            f.write(f"源目录文件数: {analysis['source_count']}\n")
            f.write(f"目标目录文件数: {analysis['target_count']}\n")
            f.write(f"缺失文件数: {analysis['missing_count']}\n")
            f.write(f"成功复制: {results['successful_copies']}\n")
            f.write(f"复制失败: {results['failed_copies']}\n")
            f.write(f"同步完成率: {results['successful_copies'] / max(analysis['missing_count'], 1) * 100:.1f}%\n\n")

            if results['copied_files']:
                f.write("成功复制的文件:\n")
                for file_path in results['copied_files']:
                    f.write(f"  - {Path(file_path).name}\n")

            if results['failed_files']:
                f.write("\n复制失败的文件:\n")
                for failed in results['failed_files']:
                    f.write(f"  - {Path(failed['file']).name}: {failed['error']}\n")

        print(f"同步报告已生成: {report_path}")

    def compare_directories_detailed(self) -> Dict:
        """
        详细比较两个目录的内容
        """
        # 获取所有文件（不限制扩展名）
        all_source_files = list(self.source_dir.glob('*'))
        all_target_files = list(self.target_dir.glob('*'))

        # 过滤出文件（排除目录）
        source_files = [f for f in all_source_files if f.is_file()]
        target_files = [f for f in all_target_files if f.is_file()]

        source_names = {f.name for f in source_files}
        target_names = {f.name for f in target_files}

        return {
            'source_only': source_names - target_names,
            'target_only': target_names - source_names,
            'common_files': source_names & target_names,
            'source_files_count': len(source_files),
            'target_files_count': len(target_files),
            'unique_source_count': len(source_names - target_names),
            'unique_target_count': len(target_names - source_names),
            'common_count': len(source_names & target_names)
        }

if __name__ == '__main__':
    # copyor = FileSynchronizer(
    #     target_dir=r'/home/ff/myProject/KGT/myProjects/myProjects/zxCodes/localPeppers/dataset/data02/test/labels',
    #     source_dir=r'/home/ff/myProject/KGT/myProjects/myProjects/zxCodes/localPeppers/dataset/detect_data03/test/labels'
    # )
    # copyor.sync_files(
    #     file_extensions=['.txt'],
    #     dry_run=False
    # )
    pass