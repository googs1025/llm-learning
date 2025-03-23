"""
# 下载脚本
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='/root/data', revision='master')
print(model_dir, "ok")

# 转换命令(必须要有 llama.cpp 项目)
python3 llama.cpp/convert_hf_to_gguf.py  /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --outfile /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ds-qw-7b.gguf --outtype f16
"""



import argparse
import subprocess
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer

def str2bool(v):
    """
    将字符串参数转换为布尔值。
    :param v: 输入的字符串
    :return: 转换后的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        return model_dir
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        return None


def convert_model_to_gguf(model_dir, outfile, outtype):
    """
    使用 llama.cpp convert_hf_to_gguf.py 转换模型格式。
    :param model_dir: 模型目录路径
    :param outfile: 输出文件路径
    :param outtype: 输出类型（例如 'f16'）
    """
    try:
        # 构建命令
        command = [
            'python3', 'llama.cpp/convert_hf_to_gguf.py',
            model_dir,
            '--outfile', outfile,
            '--outtype', outtype
        ]

        # 执行命令
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("模型转换成功！")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"模型转换过程中出现错误: {e.stderr.decode()}")
        raise


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="下载并缓存模型，并将其转换为 GGUF 格式")

    # 添加命令行参数
    parser.add_argument('--model_name', type=str, required=True,
                        help='模型的名称或路径')
    parser.add_argument('--cache_dir', type=str, default='/root/data',
                        help='缓存目录路径 (默认: /root/data)')
    parser.add_argument('--revision', type=str, default='master',
                        help='模型的版本或分支 (默认: master)')
    parser.add_argument('--outfile', type=str, required=True,
                        help='输出文件路径')
    parser.add_argument('--outtype', type=str, default='f16',
                        help='输出类型 (默认: f16)')
    parser.add_argument('--convert_to_gguf', type=str2bool, nargs='?', const=True, default=True,
                        help='是否转换为 GGUF 格式，默认为 True (可选 true/false)')
    # 解析命令行参数
    args = parser.parse_args()

    # 下载模型
    model_dir = download_model(args.model_name, args.cache_dir, args.revision)

    if model_dir and args.convert_to_gguf:
        # 确保当 --convert_to_gguf 被设置为 True 时，--outfile 参数也被提供
        if not args.outfile:
            parser.error("--outfile 是必需的当 --convert_to_gguf 设置为 True 时")
        # 转换模型
        convert_model_to_gguf(model_dir, args.outfile, args.outtype)
    elif not args.convert_to_gguf:
        print("跳过模型转换步骤。")

    # 使用方法用例
    # --model_name：必需参数，指定要下载的模型名称或路径。 ex: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # 详细查阅：https://huggingface.co/
    # --cache_dir: 可选参数，默认值为 /root/data，指定用于存储模型的缓存目录。
    # --revision: 可选参数，默认值为 master，指定模型的具体版本或分支。
    # --outfile: 必需参数，指定转换后模型的输出文件路径。
    # --outtype: 可选参数，默认值为 f16，指定输出类型。
    # --convert_to_gguf: 可选参数，默认值为 true。
    # python3 download_model_from_huggingface.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache_dir /root/data --revision your-revision-tag
    # python3 download_model_from_huggingface.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache_dir /root/data --revision master --outfile /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ds-qw-7b.gguf --outtype f16

    # 默认转换为 GGUF 格式
    # python3 download_model_from_huggingface.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache_dir /root/data --revision master --convert_to_gguf=true --outfile /root/data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ds-qw-7b.gguf --outtype f16
    # 不转换为 GGUF 格式
    # python3 download_model_from_huggingface.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --cache_dir /root/data --revision master --convert_to_gguf=false