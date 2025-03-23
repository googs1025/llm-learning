# vllm inference

### 创建使用 vllm 推理本地模型脚本
脚本可使用 vllm 进行预训练模型推理
请确保测试机器有 GPU 设备，并 GPU 显存至少 32 GB。
```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.22.3
pip install openai==1.61.0
pip install tqdm==4.67.1
pip install transformers==4.48.2
pip install vllm==0.7.1
```

脚本调用范例
```bash
root@VM-0-2-ubuntu:/home/ubuntu# python3 vllm_inference.py --model=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --prompt='请给我一个学习大模型的计划表'
INFO 03-23 19:17:30 __init__.py:183] Automatically detected platform cuda.
WARNING 03-23 19:17:31 config.py:2368] Casting torch.bfloat16 to torch.float16.
INFO 03-23 19:17:37 config.py:526] This model supports multiple tasks: {'classify', 'generate', 'reward', 'score', 'embed'}. Defaulting to 'generate'.
INFO 03-23 19:17:37 llm_engine.py:232] Initializing a V0 LLM engine (v0.7.1) with config: model='/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', speculative_config=None, tokenizer='/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
INFO 03-23 19:17:38 cuda.py:184] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 03-23 19:17:38 cuda.py:232] Using XFormers backend.
INFO 03-23 19:17:39 model_runner.py:1111] Starting to load model /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B...
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:03<00:03,  3.98s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:07<00:00,  3.77s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:07<00:00,  3.80s/it]

INFO 03-23 19:17:47 model_runner.py:1116] Loading model weights took 14.2717 GB
INFO 03-23 19:17:49 worker.py:266] Memory profiling takes 1.20 seconds
INFO 03-23 19:17:49 worker.py:266] the current vLLM instance can use total_gpu_memory (31.74GiB) x gpu_memory_utilization (0.90) = 28.57GiB
INFO 03-23 19:17:49 worker.py:266] model weights take 14.27GiB; non_torch_memory takes 0.09GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory reserved for KV Cache is 12.80GiB.
INFO 03-23 19:17:49 executor_base.py:108] # CUDA blocks: 14984, # CPU blocks: 4681
INFO 03-23 19:17:49 executor_base.py:113] Maximum concurrency for 2048 tokens per request: 117.06x
INFO 03-23 19:17:52 model_runner.py:1435] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:17<00:00,  2.01it/s]
INFO 03-23 19:18:10 model_runner.py:1563] Graph capturing finished in 17 secs, took 0.80 GiB
INFO 03-23 19:18:10 llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 22.46 seconds
Processed prompts: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:48<00:00, 48.60s/it, est. speed input: 0.25 toks/s, output: 41.91 toks/s]
Prompt: '请给我一个学习大模型的计划表<think>\n', Think: '好的，我现在要帮用户制定一个学习大模型的计划表。用户已经提供了一个详细的计划表，分为七个阶段，每个阶段有具体的学习内容和目标。我需要先理解每个阶段的重点，然后思考如何优化或补充这个计划表。\n\n首先，计划表分为七个阶段，从基础到高级，每个阶段都有明确的学习目标和内容。我应该检查每个阶段的内容是否全面，是否有遗漏的知识点。例如，在基础阶段，用户提到了机器学习基础、概率统计、优化算法和深度学习基础。这些都是必要的，但可能还需要补充一些数学基础，比如线性代数和微积分，因为这些是机器学习和大模型的基础。\n\n接下来是预训练模型阶段，用户提到了BERT、GPT、Masked Autoencoder等模型。这部分内容比较专业，但可能需要更多的实践项目，比如训练自己的模型或进行微调。此外，还可以加入一些关于模型评估和调优的内容，比如交叉验证、超参数调优等。\n\n在模型构建与调优阶段，用户提到了NLP任务、模型调优和模型部署。这部分已经不错，但可以考虑加入更多实际项目，比如在Kaggle上参加比赛，或者使用云平台进行训练和推理，这样可以更直观地理解模型的工作流程。\n\n预训练模型的微调阶段，用户提到了下游任务、模型优化和模型评估。这部分需要更多的实战练习，比如选择不同的下游任务进行微调，然后评估模型性能的变化。此外，还可以加入模型压缩和量化的内容，了解如何在资源受限的环境中部署大模型。\n\n进阶模型构建阶段，用户提到了推荐系统、计算机视觉和强化学习。这部分已经涵盖了多个领域，但可能需要更多的深度学习内容，比如迁移学习、数据增强等。此外，强化学习可能对初学者来说比较复杂，可以考虑先从基础强化学习开始，再逐步深入。\n\n模型构建与优化阶段，用户提到了模型架构设计、模型压缩和模型调优。这部分需要更多的实践，比如使用工具如PyTorch的自动化微分，或者使用模型压缩库进行实验。模型调优可能需要更多的技巧，比如学习率调度、梯度检查等。\n\n最后是实战项目与部署阶段，用户提到了项目实战和模型部署。这部分已经很好，但可以考虑加入更多关于模型部署的最佳实践，比如使用Docker容器化，或者使用云服务部署模型，比如AWS、Azure等。\n\n整体来看，用户提供的计划表已经很全面，但可能在某些阶段需要加入更多的实践项目和实际应用，以帮助用户更好地理解和应用所学知识。此外，可能需要补充一些必要的数学和编程基础，以便用户能够更深入地理解大模型的工作原理。\n\n另外，考虑到用户可能有不同的学习背景，计划表中可能需要提供一些可选的学习资源，比如推荐的书籍、在线课程或实践平台。例如，在基础阶段，可以推荐Andrew Ng的机器学习课程，或者《机器学习实战》作为入门资料。\n\n最后，计划表中的每周学习时间可以再详细一些，比如在每个阶段中安排更多的实践任务，或者在每个任务中分配更多的时间用于项目完成。这样可以帮助用户更系统地掌握知识，并在实践中提升技能。\n\n总结一下，我需要优化计划表，使其更加详细和实用，同时提供更多的学习资源和实践机会，帮助用户更好地学习大模型。\n', Answer: '\n\n### 学习大模型的详细计划表\n\n为了帮助你系统地学习大模型技术，我们制定了一份详细的学习计划表。这份计划表分为七个阶段，每个阶段都有明确的学习目标和内容，同时包含了每周的学习任务和实践项目。计划表涵盖从基础到进阶的内容，结合了理论学习和实践应用，帮助你逐步掌握大模型的核心知识和技能。\n\n---\n\n### **阶段一：基础知识入门**\n\n**目标：**\n- 掌握机器学习和深度学习的基础知识。\n- 学习必要的数学工具和编程技能。\n\n#### **每周学习任务**\n\n1. **机器学习基础**\n   - 掌握监督学习、无监督学习、强化学习的基本概念。\n   - 学习线性回归、逻辑回归、决策树等算法。\n   - 推荐书籍：《机器学习实战》（Python实现）\n   - 实践项目：使用 scikit-learn 库实现简单的分类任务。\n\n2. **概率统计基础**\n   - 学习概率分布（高斯分布、贝叶斯分布等）、贝叶斯定理。\n   - 掌握统计推断、假设检验的基本概念。\n   - 推荐课程：Coursera 《统计学习》（统计学基础）\n   - 实践项目：分析一个实际数据集，计算其统计显著性。\n\n3. **优化算法**\n   - 学习梯度下降、动量法、Adam 等优化算法。\n   - 推荐文章：Understanding the Optimization Algorithms used in Deep Learning\n   - 实践项目：实现一个简单的神经网络，并观察不同优化算法的收敛速度。\n\n4. **深度学习基础**\n   - 学习神经网络的基本结构、激活函数、前向传播和反向传播。\n   - 推荐书籍：《深度学习》（ Ian Goodfellow 等著）\n   - 实践项目：使用 TensorFlow 实现一个简单的神经网络分类器。\n\n5. **数学基础**\n   - 学习线性代数（向量、矩阵运算）、微积分（导数、梯度）。\n   - 推荐课程：Khan Academy 的线性代数与微积分课程\n   - 实践项目：推导一个常用算法的数学公式，并用代码实现。\n\n---\n\n### **阶段二：预训练模型基础**\n\n**目标：**\n- 掌握预训练模型的基本原理和架构。\n- 学习如何使用预训练模型进行下游任务。\n\n#### **每周学习任务**\n\n1. **预训练模型概述**\n   - 了解预训练模型的概念、训练过程和下游任务应用。\n   - 推荐文章：What is Pre-Training in NLP?\n   - 实践项目：使用 Hugging Face 的 Transformers 库查看预训练模型的结构。\n\n2. **BERT 系列模型**\n   - 学习BERT、BERT Fine-tuning等技术。\n   - 推荐教程：BERT 基础知识与应用\n   - 实践项目：在 Kaggle 的 NLP 比赛中使用BERT 进行文本分类。\n\n3. **GPT 系列模型**\n   - 了解 GPT、GPT-2、GPT-3 的原理和应用。\n   - 推荐文章：A Simple Explanation of How GPT-3 Works\n   - 实践项目：使用 OpenGPT 进行文本生成任务。\n\n4. **Masked Autoencoder**\n   - 学习MAE 的原理和在预训练中的应用。\n   - 推荐论文：Masked Autoencoders Are Scalable Image Pre-Conditioners\n   - 实践项目：实现一个简单的MAE模型，并观察其效果。\n\n5. **预训练模型调优**\n   - 学习超参数调优、模型微调等技术。\n   - 推荐教程：Hyperparameter Tuning for Pre-trained Models\n   - 实践项目：对一个预训练模型进行微调，评估其下游任务性能。\n\n---\n\n### **阶段三：模型构建与调优**\n\n**目标：**\n- 掌握大模型的构建和调优技巧。\n- 学习如何优化模型性能。\n\n#### **每周学习任务**\n\n1. **NLP 任务构建**\n   - 学习文本分类、命名实体识别、问答系统等任务。\n   - 推荐书籍：《自然语言处理一百个 essential tasks》\n   - 实践项目：构建一个完整的 NLP 管线，包括数据预处理、模型训练和评估。\n\n2. **模型调优**\n   - 学习模型调优方法：学习率调度、正则化、早停技术。\n   - 推荐文章：The 7 Effective Model Tuning Methods Every ML Engineer Should Know\n   - 实践项目：对一个模型进行调优，观察性能提升。\n\n3. **模型部署**\n   - 学习模型部署的基本知识和工具。\n   - 推荐教程：How to Deploy Machine Learning Models in Production\n   - 实践项目：使用 Flask 或 FastAPI 部署一个简单的 NLP 模型。\n\n4. **模型压缩与量化**\n   - 学习模型压缩技术：剪枝、量化、知识蒸馏。\n   - 推荐文章：Quantization and Model Compression for Production Deployment\n   - 实践项目：对一个大模型进行量化和压缩，评估性能变化。\n\n5. **模型解释性**\n   - 学习模型解释性技术：LIME、SHAP。\n   - 推荐教程：Interpreting and Explaining Your Models\n   - 实践项目：对模型输出进行解释性分析，可视化特征重要性。\n\n---\n\n### **阶段四：进阶模型构建**\n\n**目标：**\n- 掌握更复杂的模型架构和应用。\n- 学习如何结合多个技术实现复杂任务。\n\n#### **每周学习任务**\n\n1. **推荐系统模型**\n   - 学习协同过滤、深度学习推荐系统。\n   - 推荐文章：Deep Learning for Recommender Systems\n   - 实践项目：构建一个简单的推荐系统，并评估其性能。\n\n2. **计算机视觉模型**\n   - 学习卷积神经网络、迁移学习在 CV 中的应用。\n   - 推荐书籍：《Deep Learning for'
[rank0]:[W323 19:18:59.141406155 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
root@VM-0-2-ubuntu:/home/ubuntu#

```
### 创建兼容 OpenAI API 接口的服务器
DeepSeek-R1-Distill-Qwen 兼容 OpenAI API 协议，所以我们可以直接使用 vLLM 创建 OpenAI API 服务器。vLLM 部署实现 OpenAI API 协议的服务器非常方便。默认会在 http://localhost:8000 启动服务器。服务器当前一次托管一个模型，并实现列表模型、completions 和 chat completions 端口。

completions：是基本的文本生成任务，模型会在给定的提示后生成一段文本。这种类型的任务通常用于生成文章、故事、邮件等。
chat completions：是面向对话的任务，模型需要理解和生成对话。这种类型的任务通常用于构建聊天机器人或者对话系统。
在创建服务器时，我们可以指定模型名称、模型路径、聊天模板等参数。

- --host 和 --port 参数指定地址。
- --model 参数指定模型名称。
- --chat-template 参数指定聊天模板。
- --served-model-name 指定服务模型的名称。
- --max-model-len 指定模型的最大长度。

```python
python3 -m vllm.entrypoints.openai.api_server   --model /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   --served-model-name DeepSeek-R1-Distill-Qwen-7B   --max-model-len=2048   --dtype=half   --host 0.0.0.0   --port 8080
```

```bash
root@VM-0-2-ubuntu:/home/ubuntu# python3 -m vllm.entrypoints.openai.api_server   --model /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   --served-model-name DeepSeek-R1-Distill-Qwen-7B   --max-model-len=2048   --dtype=half   --host 0.0.0.0   --port 8080
INFO 03-23 20:10:08 __init__.py:183] Automatically detected platform cuda.
INFO 03-23 20:10:09 api_server.py:838] vLLM API server version 0.7.1
INFO 03-23 20:10:09 api_server.py:839] args: Namespace(host='0.0.0.0', port=8080, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, enable_reasoning=False, reasoning_parser=None, tool_call_parser=None, tool_parser_plugin='', model='/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', task='auto', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='half', kv_cache_dtype='auto', max_model_len=2048, guided_decoding_backend='xgrammar', logits_processor_pattern=None, distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=None, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=['DeepSeek-R1-Distill-Qwen-7B'], qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', generation_config=None, override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False)
INFO 03-23 20:10:09 api_server.py:204] Started engine process with PID 82107
WARNING 03-23 20:10:09 config.py:2368] Casting torch.bfloat16 to torch.float16.
INFO 03-23 20:10:13 __init__.py:183] Automatically detected platform cuda.
WARNING 03-23 20:10:14 config.py:2368] Casting torch.bfloat16 to torch.float16.
INFO 03-23 20:10:15 config.py:526] This model supports multiple tasks: {'generate', 'embed', 'score', 'reward', 'classify'}. Defaulting to 'generate'.
INFO 03-23 20:10:20 config.py:526] This model supports multiple tasks: {'embed', 'score', 'reward', 'generate', 'classify'}. Defaulting to 'generate'.
INFO 03-23 20:10:20 llm_engine.py:232] Initializing a V0 LLM engine (v0.7.1) with config: model='/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', speculative_config=None, tokenizer='/root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=DeepSeek-R1-Distill-Qwen-7B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=True,
INFO 03-23 20:10:21 cuda.py:184] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 03-23 20:10:21 cuda.py:232] Using XFormers backend.
INFO 03-23 20:10:22 model_runner.py:1111] Starting to load model /root/autodl-tmp/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B...
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:03<00:03,  4.00s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:07<00:00,  3.51s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:07<00:00,  3.58s/it]

INFO 03-23 20:10:29 model_runner.py:1116] Loading model weights took 14.2717 GB
INFO 03-23 20:10:31 worker.py:266] Memory profiling takes 1.19 seconds
INFO 03-23 20:10:31 worker.py:266] the current vLLM instance can use total_gpu_memory (31.74GiB) x gpu_memory_utilization (0.90) = 28.57GiB
INFO 03-23 20:10:31 worker.py:266] model weights take 14.27GiB; non_torch_memory takes 0.09GiB; PyTorch activation peak memory takes 1.40GiB; the rest of the memory reserved for KV Cache is 12.80GiB.
INFO 03-23 20:10:31 executor_base.py:108] # CUDA blocks: 14984, # CPU blocks: 4681
INFO 03-23 20:10:31 executor_base.py:113] Maximum concurrency for 2048 tokens per request: 117.06x
INFO 03-23 20:10:34 model_runner.py:1435] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 35/35 [00:17<00:00,  1.97it/s]
INFO 03-23 20:10:52 model_runner.py:1563] Graph capturing finished in 18 secs, took 0.80 GiB
INFO 03-23 20:10:52 llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 22.80 seconds
INFO 03-23 20:10:53 api_server.py:754] Using supplied chat template:
INFO 03-23 20:10:53 api_server.py:754] None
INFO 03-23 20:10:53 launcher.py:19] Available routes are:
INFO 03-23 20:10:53 launcher.py:27] Route: /openapi.json, Methods: HEAD, GET
INFO 03-23 20:10:53 launcher.py:27] Route: /docs, Methods: HEAD, GET
INFO 03-23 20:10:53 launcher.py:27] Route: /docs/oauth2-redirect, Methods: HEAD, GET
INFO 03-23 20:10:53 launcher.py:27] Route: /redoc, Methods: HEAD, GET
INFO 03-23 20:10:53 launcher.py:27] Route: /health, Methods: GET
INFO 03-23 20:10:53 launcher.py:27] Route: /ping, Methods: POST, GET
INFO 03-23 20:10:53 launcher.py:27] Route: /tokenize, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /detokenize, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/models, Methods: GET
INFO 03-23 20:10:53 launcher.py:27] Route: /version, Methods: GET
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/embeddings, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /pooling, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /score, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/score, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /rerank, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v1/rerank, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /v2/rerank, Methods: POST
INFO 03-23 20:10:53 launcher.py:27] Route: /invocations, Methods: POST
INFO:     Started server process [82050]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)


INFO:     127.0.0.1:59702 - "GET /v1/models HTTP/1.1" 200 OK
INFO 03-23 20:11:43 logger.py:37] Received request cmpl-0d0a073d89e346ef9b440c91d420f9ad-0: prompt: '我想问你，5的阶乘是多少？<think>\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=1024, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: [151646, 104100, 56007, 56568, 3837, 20, 9370, 99736, 100252, 111558, 11319, 151648, 198], lora_request: None, prompt_adapter_request: None.
INFO 03-23 20:11:43 engine.py:273] Added request cmpl-0d0a073d89e346ef9b440c91d420f9ad-0.
INFO 03-23 20:11:48 metrics.py:453] Avg prompt throughput: 2.6 tokens/s, Avg generation throughput: 37.3 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO:     127.0.0.1:59812 - "POST /v1/completions HTTP/1.1" 200 OK


INFO 03-23 20:11:59 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 3.1 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:09 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:32 logger.py:37] Received request cmpl-62e1ee2a52ea40af979816d651f626ae-0: prompt: '我想问一下大模型学习计画？<think>\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=1024, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: [151646, 104100, 56007, 100158, 26288, 104949, 100134, 37643, 54623, 11319, 151648, 198], lora_request: None, prompt_adapter_request: None.
INFO 03-23 20:12:32 engine.py:273] Added request cmpl-62e1ee2a52ea40af979816d651f626ae-0.
INFO 03-23 20:12:34 metrics.py:453] Avg prompt throughput: 2.4 tokens/s, Avg generation throughput: 17.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:39 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 42.9 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:44 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 42.3 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.2%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:49 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 42.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%.
INFO 03-23 20:12:54 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 42.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.4%, CPU KV cache usage: 0.0%.
INFO:     127.0.0.1:39544 - "POST /v1/completions HTTP/1.1" 200 OK
INFO 03-23 20:13:06 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 7.3 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 03-23 20:13:16 metrics.py:453] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.

```