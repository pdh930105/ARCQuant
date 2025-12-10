import torch
import argparse
import sys
import os
# 假设 model_utils 和之前的加载逻辑保持不变...
from model_utils import reorder_model_llama, reorder_model_qwen

import time

def get_llama(model_path):
    """加载Llama模型"""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    from transformers import LlamaForCausalLM
    # 使用 AutoTokenizer 来自动加载正确的 tokenizer 类型
    from transformers import AutoTokenizer
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # 关键修改：使用 AutoTokenizer，并明确关闭 legacy 模式
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    
    model.seqlen = 2048
    return model, tokenizer

def get_qwen(model_path):
    """加载Qwen模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_quantized_model(model_path, model_name, kv_cache=False, device='cuda:0', dataset='wikitext2', metric='max'):
    """加载并量化模型"""
    # 确定模型类型
    if "llama" in model_path.lower():
        model, tokenizer = get_llama(model_path)
        reorder_model_func = reorder_model_llama
    elif "qwen" in model_path.lower():
        model, tokenizer = get_qwen(model_path)
        reorder_model_func = reorder_model_qwen
    else:
        raise ValueError(f"Unsupported model type: {model_path}")
    
    model.eval()
    
    # 加载量化的索引文件
    import os
    dataset_name = dataset.lower()
    index_filename = f'./saved/{model_name.lower()}_reorder_index_{dataset_name}_{metric}.pt'
    select_num_filename = f'./saved/{model_name.lower()}_select_num_{dataset_name}_{metric}.pt'
        
    if not os.path.isfile(index_filename):
        raise FileNotFoundError(f"Cannot find reorder index file for {model_name}. "
                              f"Expected at: {index_filename}")
    
    print("Loading cached reordering index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    select_nums = torch.load(select_num_filename, weights_only=False)
    
    torch.cuda.reset_max_memory_allocated()
    print("Reordering and quantizing model...")
    start_time=time.time()
    model = reorder_model_func(
        model, device=device, kv_cache=kv_cache, reorder_index=reorder_index, select_nums=select_nums
    )
    end_time=time.time()
    # 将模型移到GPU
    model.to(device)
    peak_memory = torch.cuda.max_memory_allocated()

    print(model)
    print(f"Quantized Model Size: {peak_memory/(1024*1024*1024):.2f} GB")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return model, tokenizer


# ==========================================
# 新增/修改的核心部分
# ==========================================

class ChatSession:
    """管理对话历史和生成的类"""
    def __init__(self, model, tokenizer, device='cuda:0'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = []  # 存储上下文 [{"role": "user", "content": "..."}, ...]

    def clear_history(self):
        self.history = []
        print("\n[系统] 上下文已清空。\n")

    def generate(self, user_input, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        # 1. 更新用户输入到历史
        self.history.append({"role": "user", "content": user_input})

        # 2. 生成 Prompt 字符串 (不直接转 Tensor，防止报错)
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # 如果历史记录脏了，回退到单轮对话，防止死循环
            print(f"\n[Warning] 上下文格式错误，重置记忆... Error: {e}")
            self.history = [{"role": "user", "content": user_input}]
            prompt_text = self.tokenizer.apply_chat_template(
                self.history, tokenize=False, add_generation_prompt=True
            )

        # 3. 手动分词 (add_special_tokens=False 因为模板里已经有了)
        input_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        
        input_length = input_ids.input_ids.shape[1]

        # 4. 生成配置
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # 5. 推理
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                **generation_config
            )

        # 6. 解码 (保留特殊字符，以便我们手动处理)
        response_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        # =====================================================
        # [核心修复] 清洗所有可能导致下一轮报错的特殊结束标记
        # =====================================================
        
        # 定义需要移除的“坏”标记列表
        bad_tokens = [
            # Llama 3.1 特有
            "<|eot_id|>", 
            "<|eom_id|>", 
            "<|start_header_id|>", 
            "<|end_header_id|>",
            # Qwen 特有
            "<|im_end|>", 
            "<|im_start|>",
            # 通用
            "<|endoftext|>",
            self.tokenizer.eos_token  # 自动获取当前模型的 EOS
        ]
        
        # 过滤掉 None 类型 (防止 self.tokenizer.eos_token 为 None)
        bad_tokens = [t for t in bad_tokens if t is not None]

        # 执行清洗
        for token in bad_tokens:
            response = response.replace(token, "")

        response = response.strip()

        # 7. 更新历史
        self.history.append({"role": "assistant", "content": response})

        return response

def interactive_mode(model, tokenizer, device='cuda:0'):
    """交互模式 (支持多轮对话)"""
    session = ChatSession(model, tokenizer, device)
    
    print("\n" + "="*50)
    print(f"模型加载完成！当前设备: {device}")
    print("指令说明:")
    print("  - 输入对话内容按回车发送")
    print("  - 输入 '/clear' 清空上下文记忆")
    print("  - 输入 '/quit' 或 'exit' 退出")
    print("="*50 + "\n")
    
    while True:
        try:
            prompt = input("User >>> ").strip()
            
            # 处理特殊指令
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            if prompt.lower() == '/clear':
                session.clear_history()
                continue
            if not prompt:
                continue
            
            print("Bot  >>> 思考中...", end="", flush=True)
            
            # 生成回复
            # 针对 Qwen3，由于可能有 <think> 过程，建议适当增加 max_new_tokens
            response = session.generate(
                prompt,
                max_new_tokens=1024,  # Qwen3 思考模式可能需要更长的长度
                temperature=0.7, 
                top_p=0.9
            )
            
            # 清除 "思考中..." 并打印结果
            print(f"\rBot  >>> {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n[系统] 检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n[错误] 生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 出错时不中断循环，允许重试
            continue

def single_prompt_mode(model, tokenizer, prompt, device='cuda:0'):
    """单次模式也使用 Session 以复用逻辑"""
    session = ChatSession(model, tokenizer, device)
    print(f"User: {prompt}")
    print("\n生成中...")
    response = session.generate(prompt, max_new_tokens=1024)
    print(f"Bot: {response}")

# ==========================================
# Main 函数 (稍微调整调用逻辑)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="量化模型多轮对话Demo")
    
    # 必需参数
    parser.add_argument('model_path', type=str, help='模型路径')
    
    # 可选参数
    parser.add_argument('--prompt', type=str, default=None, help='单次模式提示词')
    parser.add_argument('--kv_cache', action='store_true', help='是否量化KV缓存')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备')
    parser.add_argument(
        "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "humaneval"], 
        help="校准数据集"
    )
    parser.add_argument(
        '--act_sort_metric', type=str, default='max', choices=['mean', 'frobenius', 'hessian', 'max'],
        help='通道重排策略'
    )
    
    args = parser.parse_args()
    
    try:
        model_name = args.model_path.split('/')[-2] if '/' in args.model_path else args.model_path
        
        print(f"正在加载: {model_name} ...")
        
        # 调用你原有的加载函数
        model, tokenizer = load_quantized_model(
            args.model_path, 
            model_name, 
            kv_cache=args.kv_cache,
            device=args.device,
            dataset=args.dataset,
            metric=args.act_sort_metric
        )
        
        if args.prompt:
            single_prompt_mode(model, tokenizer, args.prompt, args.device)
        else:
            interactive_mode(model, tokenizer, args.device)
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()