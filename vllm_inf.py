from vllm import LLM, SamplingParams
import math
import torch

# グローバル変数としてモデルとトークナイザーを定義
llm = None
tokenizer = None


"""
LLMとトークナイザーを初期化する関数
"""
llm, tokenizer

# GPUが7つあることを確認
num_gpus = torch.cuda.device_count()
exponent = math.floor(math.log2(num_gpus))
tensor_parallel_gpu = 2**exponent

print(f"利用可能なGPUは{num_gpus}個です。よって{tensor_parallel_gpu}個のGPUを使用します。")
    
llm = LLM(
    model="team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-AWQ",
    quantization="awq",
    gpu_memory_utilization=0.9,  # 必要に応じて調整
    tensor_parallel_size=tensor_parallel_gpu  # アテンションヘッド数(32)の約数を指定
)
tokenizer = llm.get_tokenizer()
print("LLMモデルとトークナイザーが初期化されました。")


SYSTEM_PROMPT = "あなたは優秀な日本語エージェントです。"

def vllm_chat(user_inputs, system_prompt = SYSTEM_PROMPT):
    """
    複数のユーザー入力に対して、vLLMでバッチ推論を実行する関数。
    モデルはグローバル変数としてロード済み。
    
    Args:
        user_inputs_list (list): ユーザー入力のリスト。
        system_prompt (str): システムプロンプト。
    
    Returns:
        list: AIの応答のリスト。
    """
    global llm, tokenizer
    if llm is None or tokenizer is None:
        raise RuntimeError("LLMモデルが初期化されていません。initialize_llm() を実行してください。")
    
    if type(user_inputs) == str:
        input_messages_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_inputs},
        ]
    elif type(user_inputs) == list:
        input_messages_list = user_inputs.copy()  # コピーを作成
        input_messages_list.insert(0, {"role": "system", "content": system_prompt})
    else:
        raise ValueError(
            "user_inputs should be a string or a list."
        )  # エラーハンドリングを追加
    
    prompt = tokenizer.apply_chat_template(
        input_messages_list,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4092
    )

    outputs = llm.generate(prompt, sampling_params)
    # Extract and print only the AI's response
    return (outputs[0].outputs[0].text)

def vllm_chat_batch(user_inputs_list, system_prompt):
    """
    複数のユーザー入力に対して、vLLMでバッチ推論を実行する関数。
    モデルはグローバル変数としてロード済み。
    
    Args:
        user_inputs_list (list): ユーザー入力のリスト。
        system_prompt (str): システムプロンプト。
    
    Returns:
        list: AIの応答のリスト。
    """
    
    global llm, tokenizer
    if llm is None or tokenizer is None:
        raise RuntimeError("LLMモデルが初期化されていません。initialize_llm() を実行してください。")
    
    prompts = []

    for user_inputs in user_inputs_list:
        if type(user_inputs) == str:
            input_messages_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_inputs},
            ]
        elif type(user_inputs) == list:
            input_messages_list = user_inputs.copy()  # コピーを作成
            input_messages_list.insert(0, {"role": "system", "content": system_prompt})
        else:
            raise ValueError(
                "user_inputs should be a string or a list."
            )  # エラーハンドリングを追加

        prompt = tokenizer.apply_chat_template(
            input_messages_list,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=1000
    )

    outputs = llm.generate(prompts, sampling_params)
    
    ai_responses = [output.outputs[0].text for output in outputs]
    
    return ai_responses