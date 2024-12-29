# vLLM Japanese LLM Interface

このプロジェクトは、vLLMを使用して日本語LLMモデルを効率的に実行するためのインターフェースを提供します。

## 特徴

- vLLMを使用した高速な推論処理
- チャット形式での対話サポート
- 複数GPUの自動検出と最適な並列処理の設定
- バッチ処理による効率的な推論
- AWQ量子化モデルのサポート

## 必要条件

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本的な使用方法

```python
from vllm_inf import vllm_chat

# 単一の質問
response = vllm_chat("こんにちは")
print(response)

# チャット履歴を使用した対話
chat_history = [
    {"role": "user", "content": "こんにちは"},
    {"role": "assistant", "content": "こんにちは！お手伝いできることはありますか？"},
    {"role": "user", "content": "今日の天気を教えてください"}
]
response = vllm_chat(chat_history)
print(response)
```

### バッチ処理

```python
from vllm_inf import vllm_chat_batch

# 複数の質問を一括処理
questions = [
    "こんにちは",
    "今日の天気は？",
    "おすすめの本を教えて"
]
responses = vllm_chat_batch(questions, system_prompt="あなたは優秀な日本語エージェントです。")
for response in responses:
    print(response)
```

## 主な機能

### vllm_chat()
- 単一のユーザー入力または会話履歴に対して応答を生成
- システムプロンプトのカスタマイズが可能
- 文字列または会話履歴リストを入力として受け付け

### vllm_chat_batch()
- 複数の入力に対して一括で応答を生成
- 効率的なバッチ処理による高速な推論
- 同一のシステムプロンプトを使用

## 設定

- デフォルトのシステムプロンプト: "あなたは優秀な日本語エージェントです。"
- 推論パラメータ:
  - temperature: 0.6
  - top_p: 0.9
  - max_tokens: 4092（単一応答）/ 1000（バッチ処理）

## GPU利用

- 利用可能なGPU数を自動検出
- 最適なテンソル並列サイズを自動計算
- AWQ量子化による効率的なメモリ使用

## 注意事項

- GPUメモリ使用率は0.9に設定されています（必要に応じて調整可能）
- モデルは初回実行時に自動的に初期化されます
- バッチ処理時は、メモリ使用量に注意してください
