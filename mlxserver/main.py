import asyncio
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, List

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from mlx_lm import load, generate

from . import rag_manager

# --- 1. 設定管理 ---
def load_config() -> dict:
    """設定ファイル(~/.my_mlx_server/config.json)を読み込む"""
    config_file = Path.home() / ".my_mlx_server" / "config.json"
    if not config_file.exists():
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

config = load_config()

class Settings(BaseSettings):
    """アプリケーションの設定を管理するクラス"""
    models_dir: Path = Path(config.get("models_dir", str(Path.home() / "my_mlx_models")))
    personalities_dir: Path = Path.home() / ".my_mlx_server" / "personalities"
    model_ttl: int = int(config.get("model_ttl", 180))
    max_loaded_models: int = int(config.get("max_loaded_models", 1))
    api_key: Optional[str] = config.get("api_key", None)
    port: int = int(config.get("port", 27000))

    class Config:
        env_file = ".env"

# --- 2. グローバル変数と初期化 ---
settings = Settings()
app = FastAPI(
    title="Custom MLX Model Server",
    description="Ollama-compatible API server for MLX models with auto-management.",
    version="1.1.0",
)

loaded_models: dict[str, dict[str, Any]] = {}
loading_locks: dict[str, asyncio.Lock] = {}

# 起動時に必要なディレクトリを作成
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.personalities_dir.mkdir(parents=True, exist_ok=True)

# サーバー起動時の設定内容をコンソールに表示
print("-" * 50)
print("MLX Server is starting up with the following settings:")
print(f"  - Models Directory: '{settings.models_dir}'")
print(f"  - Personalities Directory: '{settings.personalities_dir}'")
print(f"  - Listen Port: {settings.port}")
print(f"  - Max Concurrent Loaded Models: {settings.max_loaded_models}")
print(f"  - Model Idle Timeout (TTL): {settings.model_ttl} seconds")
print(f"  - API Key Protection: {'Enabled' if settings.api_key else 'Disabled'}")
print("-" * 50)

# --- 3. APIキー認証 ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """APIキーを検証するFastAPIの依存性注入関数"""
    if settings.api_key:
        if api_key is None:
            raise HTTPException(status_code=403, detail="API Key required. Please provide it in the 'X-API-Key' header.")
        if api_key != settings.api_key:
            raise HTTPException(status_code=403, detail="Invalid API Key.")
    return True

# --- 4. バックグラウンドタスク ---
async def cleanup_idle_models():
    """アイドル状態のモデルを定期的にアンロードするバックグラウンドタスク"""
    while True:
        await asyncio.sleep(30)
        now = datetime.utcnow()
        idle_threshold = timedelta(seconds=settings.model_ttl)
        
        models_to_unload = [
            name for name, cache in loaded_models.items()
            if (now - cache["last_used"]) > idle_threshold
        ]
        
        for name in models_to_unload:
            if name in loaded_models:
                print(f"Model '{name}' has been idle and will be unloaded.")
                del loaded_models[name]
                if name in loading_locks:
                    del loading_locks[name]
                gc.collect()

@app.on_event("startup")
async def startup_event():
    """FastAPIサーバー起動時にバックグラウンドタスクを開始する"""
    asyncio.create_task(cleanup_idle_models())

# --- 5. Pydanticモデル (Ollama互換) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., description="使用するベースモデル名または個性名")
    messages: List[Message]
    stream: bool = False

class ChatResponseMessage(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    model: str
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: ChatResponseMessage
    done: bool

class EmbeddingsRequest(BaseModel):
    model_name: str
    input: List[str]

class EmbeddingsData(BaseModel):
    embedding: List[float]
    index: int

class EmbeddingsResponse(BaseModel):
    model: str
    data: List[EmbeddingsData]

# --- 6. APIエンドポイント ---
@app.get("/", summary="サーバーの状態を確認")
async def root():
    return {"message": "MLX Auto-Managed Server is running."}

@app.get("/api/tags", summary="利用可能なモデルと個性を一覧表示 (Ollama互換)")
async def list_tags():
    models = [{"name": f"{d.name}:latest", "model": f"{d.name}:latest"} for d in settings.models_dir.iterdir() if d.is_dir()]
    personalities = []
    for f in settings.personalities_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as pf:
                data = json.load(pf)
                # 個性は、個性名とベースモデル名の両方でタグとして表示
                personalities.append({"name": f"{f.stem}:latest", "model": f"{f.stem}:latest"})
        except Exception:
            continue
    return {"models": models + personalities}

@app.post("/api/embeddings", response_model=EmbeddingsResponse, dependencies=[Depends(verify_api_key)])
async def create_embeddings(request: EmbeddingsRequest):
    """テキストの埋め込みベクトルを生成する (注:ダミー実装)"""
    model_name = request.model_name
    
    # 実際にはここでモデルの自動ロードが必要
    if model_name not in loaded_models:
        raise HTTPException(status_code=400, detail=f"Embedding model '{model_name}' is not loaded. Please warm it up with a chat request first.")

    cache = loaded_models[model_name]
    model = cache["model"]
    loaded_models[model_name]["last_used"] = datetime.utcnow()

    embeddings_data = []
    # (注: この部分は使用する埋め込みモデルのアーキテクチャに大きく依存します)
    #      mlx-lmの `get_embeddings` を使うか、モデルのforward passを直接呼び出す必要があります。
    #      ここでは、機能の骨格を示すためのダミー実装としてランダムなベクトルを返します。
    print(f"Generating DUMMY embeddings for {len(request.input)} items...")
    for i, text in enumerate(request.input):
        hidden_dim = getattr(model.config, 'hidden_size', 512)
        embedding_vector = list(np.random.rand(hidden_dim))
        embeddings_data.append(EmbeddingsData(embedding=embedding_vector, index=i))
    
    return EmbeddingsResponse(model=model_name, data=embeddings_data)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, authorized: bool = Depends(verify_api_key)):
    """Ollama互換のチャットエンドポイント"""
    base_model_name = request.model
    messages = request.messages

    # リクエストされたモデルが「個性」であれば、ベースモデルとシステムプロンプトを展開
    personality_file = settings.personalities_dir / f"{base_model_name}.json"
    if personality_file.exists():
        try:
            with open(personality_file, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            base_model_name = p_data.get("base_model")
            system_prompt = p_data.get("system_prompt")
            # messages配列の先頭にシステムプロンプトを追加（既にあれば上書き）
            messages = [{"role": "system", "content": system_prompt}] + [m.model_dump() for m in messages if m.role != "system"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to apply personality '{request.model}': {e}")

    # ベースモデルの自動ロード
    if base_model_name not in loaded_models:
        # メモリ上限チェックとLRUキャッシュによる解放
        if len(loaded_models) >= settings.max_loaded_models:
            oldest_model_name = min(loaded_models.items(), key=lambda item: item[1]['last_used'])[0]
            print(f"Max models loaded ({settings.max_loaded_models}). Unloading least recently used model: '{oldest_model_name}'")
            del loaded_models[oldest_model_name]
            gc.collect()
        
        if base_model_name not in loading_locks:
            loading_locks[base_model_name] = asyncio.Lock()
        
        async with loading_locks[base_model_name]:
            if base_model_name not in loaded_models:
                model_path = settings.models_dir / base_model_name
                if not model_path.exists():
                    raise HTTPException(status_code=404, detail=f"Base model '{base_model_name}' not found.")
                print(f"Auto-loading model '{base_model_name}'...")
                try:
                    model, tokenizer = load(str(model_path))
                    loaded_models[base_model_name] = {"model": model, "tokenizer": tokenizer, "last_used": datetime.utcnow()}
                    print(f"Model '{base_model_name}' loaded successfully.")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error while loading model: {e}")

    # モデル利用のタイムスタンプを更新
    loaded_models[base_model_name]["last_used"] = datetime.utcnow()
    cache = loaded_models[base_model_name]
    model, tokenizer = cache["model"], cache["tokenizer"]

    try:
        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply chat template. The base model may not support it or the message format is incorrect. Error: {e}")
    
    # ストリーミング応答
    if request.stream:
        async def stream_generator():
            token_iterator = generate(model, tokenizer, prompt=final_prompt, stream=True)
            for token_chunk, _ in token_iterator:
                response = ChatResponse(
                    model=request.model,
                    message=ChatResponseMessage(role="assistant", content=token_chunk),
                    done=False
                )
                yield f"{response.model_dump_json(exclude_none=True, ensure_ascii=False)}\n"
                await asyncio.sleep(0.01) # 短い待機でCPUを解放
            
            final_response = ChatResponse(
                model=request.model,
                message=ChatResponseMessage(role="assistant", content=""),
                done=True
            )
            yield f"{final_response.model_dump_json(exclude_none=True, ensure_ascii=False)}\n"
        return StreamingResponse(stream_generator(), media_type="application/x-ndjson")
    
    # 非ストリーミング応答
    else:
        full_response = generate(model, tokenizer, prompt=final_prompt)
        response_data = ChatResponse(
            model=request.model,
            message=ChatResponseMessage(role="assistant", content=full_response),
            done=True
        )
        return response_data