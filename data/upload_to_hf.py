# -*- coding: utf-8 -*-
"""
将自建「药板药片外观缺陷」数据集上传到 HuggingFace Datasets。

前置步骤：
    1. 安装依赖：pip install huggingface_hub
    2. 登录（二选一）：
       - 命令行登录：huggingface-cli login  （粘贴你的 Write Token）
       - 环境变量：Windows 用 `set HF_TOKEN=hf_xxx`，Linux/macOS 用 `export HF_TOKEN=hf_xxx`

用法：
    python data/upload_to_hf.py --repo-id <你的用户名>/pharma-blister-defect-dataset

可选参数：
    --private   创建私有数据集仓库（默认公开）

说明：
    - 自动创建（或复用）HuggingFace 上的 dataset 仓库；
    - 上传 data/raw（原始）、data/processed（预处理后）与数据集卡片；
    - 完成后访问 https://huggingface.co/datasets/<repo-id> 查看。

（若改用 ModelScope 魔搭：安装 `modelscope`，把 create_repo/upload 替换为
  modelscope.hub.api.HubApi 的对应接口即可，结构一致。）
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARD = HERE / "hf_dataset_card.md"


def main():
    parser = argparse.ArgumentParser(description="上传自建数据集到 HuggingFace Datasets")
    parser.add_argument("--repo-id", required=True,
                        help="HuggingFace 数据集仓库 ID，例如 <用户名>/pharma-blister-defect-dataset")
    parser.add_argument("--private", action="store_true", help="创建私有仓库")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                          private=args.private, exist_ok=True)
    print(f"仓库就绪：{url}")

    # 上传原始数据
    api.upload_folder(folder_path=str(HERE / "raw"), path_in_repo="raw",
                      repo_id=args.repo_id, repo_type="dataset")
    # 上传预处理后数据
    api.upload_folder(folder_path=str(HERE / "processed"), path_in_repo="processed",
                      repo_id=args.repo_id, repo_type="dataset")
    # 上传数据集卡片（作为仓库 README）
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset")

    print(f"\n上传完成：https://huggingface.co/datasets/{args.repo_id}")
    print("请将上述链接填入 data/README.md 与 README.md 的占位处（替换 <你的用户名>）。")


if __name__ == "__main__":
    main()
