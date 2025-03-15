# python3 llama.cpp/convert_hf_to_gguf.py  /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# --outfile /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ds-qw-7b.gguf --outtype f16



#
# import torch
# from modelscope import snapshot_download, AutoModel, AutoTokenizer
# import os
# model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='/root/data', revision='master')
# print(model_dir, "ok")


import argparse
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer


def download_model(model_name, cache_dir, revision):
    """
    下载指定的模型到本地缓存目录。

    :param model_name: 模型的名称或路径
    :param cache_dir: 缓存目录路径
    :param revision: 模型的版本或分支
    """
    try:
        model_dir = snapshot_download(model_name, cache_dir=cache_dir, revision=revision)
        print(f"模型已下载至: {model_dir}")
    except Exception as e:
        print(f"下载过程中出现错误: {e}")


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="下载并缓存模型")

    # 添加命令行参数
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型的名称或路径')
    parser.add_argument('--cache_dir', type=str, default='/root/data',
                        help='缓存目录路径 (默认: /root/data)')
    parser.add_argument('--revision', type=str, default='master',
                        help='模型的版本或分支 (默认: master)')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数下载模型
    download_model(args.model_name, args.cache_dir, args.revision)

    # 使用方法用例
    # --model_name：必需参数，指定要下载的模型名称或路径。 ex: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # 详细查阅：https://huggingface.co/
    # --cache_dir：可选参数，默认值为 /root/data，指定用于存储模型的缓存目录。
    # --revision：可选参数，默认值为 master，指定模型的具体版本或分支。
    # python download_model.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache_dir /your/custom/cache/dir --revision your-revision-tag
