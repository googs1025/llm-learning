import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE'] = 'True'


# 注：vLLM 需要使用 GPU 设备，不使用会遇到报错。

def get_completion(prompts, model, tokenizer=None, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048):
    """
    定义一个函数来获取模型生成的文本。

    参数:
    - prompts: 要处理的输入提示列表。
    - model: 模型路径或名称。
    - tokenizer: 分词器对象，用于将文本转换为token ID序列。
    - max_tokens: 生成的最大token数。
    - temperature: 控制生成文本多样性的参数，值越高多样性越大。
    - top_p: 核心采样（nucleus sampling）的概率阈值。
    - max_model_len: 模型能够处理的最大序列长度。
    """
    stop_token_ids = [151329, 151336, 151338]  # 定义停止标记ID列表
    # 创建采样参数。temperature控制生成文本的多样性，top_p控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                     stop_token_ids=stop_token_ids)
    # 初始化vLLM推理引擎，使用指定的数据类型进行推理，默认是float16以节省显存
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len, trust_remote_code=True, dtype="float16")
    outputs = llm.generate(prompts, sampling_params)  # 使用模型生成文本
    return outputs


def main():
    """
    主函数，负责解析命令行参数并调用get_completion函数生成文本。
    """
    parser = argparse.ArgumentParser(description='Generate text using a language model.')
    # 添加命令行参数
    # 用户参数
    # --model 输入模型路径：
    # model = '/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'  # 指定模型路径
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 指定模型名称，自动下载模型
    # --prompt 输入提示词：
    #
    parser.add_argument('--model', type=str, required=True, help='Path or name of the model, can support loacl model '
                                                                 'or remote model')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for the model')
    # 模型参数
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for diversity in generation')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p (nucleus) sampling probability threshold')
    parser.add_argument('--max_model_len', type=int, default=2048, help='Maximum sequence length the model can handle')

    args = parser.parse_args()  # 解析命令行参数

    tokenizer = None  # 如果需要，可以加载分词器

    # 示例提示：根据DeepSeek官方建议，每个prompt都应以<think>\n结尾
    # 例如：
    # text = [
    #    "请帮我制定个简短的初学者Python学习计划<think>\n", ]  # 可用List同时传入多个prompt

    # 这里可以构建聊天模板的消息，但不是必需的。
    # messages = [
    #     {"role": "user", "content": prompt+"<think>\n"}
    # ]
    # 如果需要应用聊天模板，可以取消下面的注释并调整相关代码
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    text = [args.prompt + "<think>\n"]  # 构建输入提示列表

    # 调用get_completion函数获取生成的文本
    outputs = get_completion(text, args.model, tokenizer=tokenizer, max_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p, max_model_len=args.max_model_len)

    # 输出是一个包含prompt、生成文本和其他信息的RequestOutput对象列表
    for output in outputs:
        prompt = output.prompt  # 获取原始输入提示
        generated_text = output.outputs[0].text  # 获取生成的文本
        if r"</think>" in generated_text:  # 判断是否包含思考过程结束标记
            think_content, answer_content = generated_text.split(r"</think>")  # 分割出思考内容和答案
        else:
            think_content = ""  # 如果没有思考过程结束标记，则思考内容为空
            answer_content = generated_text  # 将整个生成文本视为答案
        print(f"Prompt: {prompt!r}, Think: {think_content!r}, Answer: {answer_content!r}")  # 打印结果


if __name__ == "__main__":
    main()  # 调用主函数