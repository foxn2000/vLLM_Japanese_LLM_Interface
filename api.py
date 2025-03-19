# filename: main.py
from typing import List, Union

import math
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

# ==== ここからvLLM関連のコード ====
# グローバル変数としてモデルとトークナイザーを定義
llm = None
tokenizer = None

SYSTEM_PROMPT = "あなたは優秀な日本語エージェントです。"

def initialize_llm():
    """LLMとトークナイザーを初期化する関数"""
    global llm, tokenizer

    # GPUの数を取得
    num_gpus = torch.cuda.device_count()
    exponent = math.floor(math.log2(num_gpus)) if num_gpus > 0 else 0
    tensor_parallel_gpu = max(1, 2 ** exponent)

    print(f"利用可能なGPUは{num_gpus}個です。よって{tensor_parallel_gpu}個のGPUを使用します。")

    llm = LLM(
        model="team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.9,  # 必要に応じて調整
        tensor_parallel_size=tensor_parallel_gpu  # アテンションヘッド数(32)の約数を指定
    )
    tokenizer = llm.get_tokenizer()
    print("LLMモデルとトークナイザーが初期化されました。")


def vllm_chat_batch(user_inputs_list: List[Union[str, list]], system_prompt: str = SYSTEM_PROMPT):
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
        if isinstance(user_inputs, str):
            input_messages_list = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_inputs},
            ]
        elif isinstance(user_inputs, list):
            input_messages_list = user_inputs.copy()  # コピーを作成
            input_messages_list.insert(0, {"role": "system", "content": system_prompt})
        else:
            raise ValueError("user_inputs should be a string or a list.")

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

    # バッチで推論を実行
    outputs = llm.generate(prompts, sampling_params)
    ai_responses = [output.outputs[0].text for output in outputs]
    return ai_responses
# ==== ここまでvLLM関連のコード ====


# ==== FastAPI用のリクエストボディ定義 ====
class ChatBatchRequest(BaseModel):
    user_inputs_list: List[Union[str, list]]
    system_prompt: str = SYSTEM_PROMPT


# ==== アプリ起動時にLLMを初期化 ====
@app.on_event("startup")
def on_startup():
    initialize_llm()


# ==== バッチ推論用のエンドポイント ====
@app.post("/chat_batch")
def chat_batch(request: ChatBatchRequest):
    """
    user_inputs_list: ["質問1", "質問2", ...] または
                      [["role":"user","content":"..."}, ...] のリスト
    system_prompt: システムプロンプト（デフォルトは "あなたは優秀な日本語エージェントです。"）
    """
    responses = vllm_chat_batch(
        user_inputs_list=request.user_inputs_list,
        system_prompt=request.system_prompt
    )
    return {"responses": responses}
