# vLLM Japanese LLM Interface

このプロジェクトは、vLLM を使用して日本語 LLM モデルを効率的に実行するためのインターフェースを提供します。  
単一・バッチのチャット形式での推論が可能で、FastAPI を利用した API として簡単にデプロイできます。

---

## 特徴

- **vLLM を使用した高速な推論処理**  
  大規模言語モデルを効率的に動作させるために最適化。
- **チャット形式での対話サポート**  
  system / user / assistant ロールを活用した会話形式。
- **複数 GPU の自動検出と最適な並列処理の設定**  
  利用可能な GPU 数を自動検出して並列度を適切に設定。
- **バッチ処理による効率的な推論**  
  複数の問い合わせを一度に処理し、応答を生成可能。
- **AWQ 量子化モデルのサポート**  
  メモリ使用量を抑えつつ高速に動作。

---

## 必要条件

以下のコマンドで依存パッケージをインストールしてください。  
`requirements.txt` に必要なライブラリを記載している想定です。

```bash
pip install -r requirements.txt
```

必要なライブラリの例（バージョンは環境に合わせて調整してください）:
- `fastapi`
- `uvicorn`
- `vllm`
---

## 使用方法 (Python コード単体での利用)

### 単一の質問に対する推論

```python
from vllm_inf import vllm_chat

# 単一の質問
response = vllm_chat("こんにちは")
print(response)
```

### 会話履歴を利用した推論

```python
from vllm_inf import vllm_chat

# チャット履歴
chat_history = [
    {"role": "user", "content": "こんにちは"},
    {"role": "assistant", "content": "こんにちは！お手伝いできることはありますか？"},
    {"role": "user", "content": "今日の天気を教えてください"}
]
response = vllm_chat(chat_history)
print(response)
```

### バッチ処理による推論

```python
from vllm_inf import vllm_chat_batch

# 複数の質問を一括処理
questions = [
    "こんにちは",
    "今日の天気は？",
    "おすすめの本を教えて"
]
responses = vllm_chat_batch(
    questions,
    system_prompt="あなたは優秀な日本語エージェントです。"
)
for r in responses:
    print(r)
```

---

## FastAPI を利用した API サーバ

### `main.py` (例)

```python
from typing import List, Union
import math
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

# グローバル変数としてモデルとトークナイザーを定義
llm = None
tokenizer = None

SYSTEM_PROMPT = "あなたは優秀な日本語エージェントです。"

def initialize_llm():
    """LLMとトークナイザーを初期化する関数"""
    global llm, tokenizer
    num_gpus = torch.cuda.device_count()
    exponent = math.floor(math.log2(num_gpus)) if num_gpus > 0 else 0
    tensor_parallel_gpu = max(1, 2 ** exponent)

    print(f"利用可能なGPUは{num_gpus}個です。よって{tensor_parallel_gpu}個のGPUを使用します。")

    llm = LLM(
        model="team-hatakeyama-phase2/Tanuki-8B-dpo-v1.0-AWQ",
        quantization="awq",
        gpu_memory_utilization=0.9,  # 必要に応じて調整
        tensor_parallel_size=tensor_parallel_gpu
    )
    tokenizer = llm.get_tokenizer()
    print("LLMモデルとトークナイザーが初期化されました。")

def vllm_chat_batch(user_inputs_list: List[Union[str, list]], system_prompt: str = SYSTEM_PROMPT):
    """複数のユーザー入力に対して、vLLMでバッチ推論を実行する関数。"""
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
            input_messages_list = user_inputs.copy()
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

    outputs = llm.generate(prompts, sampling_params)
    ai_responses = [output.outputs[0].text for output in outputs]
    return ai_responses

# ==== FastAPI 用のリクエストボディ定義 ====
class ChatBatchRequest(BaseModel):
    user_inputs_list: List[Union[str, list]]
    system_prompt: str = SYSTEM_PROMPT

@app.on_event("startup")
def on_startup():
    initialize_llm()

@app.post("/chat_batch")
def chat_batch(request: ChatBatchRequest):
    """
    user_inputs_list: ["質問1", "質問2", ...] または
                      [["role":"user","content":"..."}, ...] のリスト
    system_prompt: システムプロンプト
    """
    responses = vllm_chat_batch(
        user_inputs_list=request.user_inputs_list,
        system_prompt=request.system_prompt
    )
    return {"responses": responses}
```

### 実行方法

1. 上記の `main.py` を準備します。  
2. 以下のコマンドで FastAPI サーバを起動します。

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. 起動後、下記のような JSON リクエストを `POST /chat_batch` に送ることでバッチ推論を行えます。

   ```json
   {
     "user_inputs_list": [
       "こんにちは！",
       "今日の天気を教えてください。"
     ]
   }
   ```

4. レスポンスとして、以下のような JSON が返却されます。

   ```json
   {
     "responses": [
       "...",  // 1つ目の質問への応答
       "..."   // 2つ目の質問への応答
     ]
   }
   ```

---

## 主な機能

### `vllm_chat()`
- 単一のユーザー入力または会話履歴に対して応答を生成
- システムプロンプトのカスタマイズが可能
- 文字列または会話履歴リストを入力として受け付ける

### `vllm_chat_batch()`
- 複数の入力に対して一括で応答を生成
- 効率的なバッチ処理による高速な推論
- 同一のシステムプロンプトを使用可能

---

## 設定

- **デフォルトのシステムプロンプト**  
  `"あなたは優秀な日本語エージェントです。"`

- **推論パラメータ**  
  - `temperature`: 0.6  
  - `top_p`: 0.9  
  - `max_tokens`: 
    - 4092（単一チャット推論時など）  
    - 1000（バッチ処理時）

- **GPU メモリ使用率**  
  `0.9` に設定 (必要に応じて調整)

---

## GPU 利用

- 利用可能な GPU 数を自動検出  
  `torch.cuda.device_count()` を用いて GPU 数を検出します。
- 最適なテンソル並列サイズを自動計算  
  `2^floor(log2(num_gpus))` に基づく並列度を自動設定。
- **AWQ 量子化** を採用し、高速かつメモリ効率を実現。

---

## 注意事項

- モデルは初回実行時に自動的に初期化されます。  
- バッチ処理時は、GPU メモリ使用量に注意してください。  
- 大規模な入力や高い `max_tokens` を指定すると、メモリを大量に消費する可能性があります。

---
