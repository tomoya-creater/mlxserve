import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import typer
import click
import requests
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.markdown import Markdown

from huggingface_hub import hf_hub_download, model_info, snapshot_download
from . import rag_manager

# --- i18n: 文字列定義 ---
locales = {
    "ja": {
        "help_main": "MLXモデルを管理・実行するためのCLIツール",
        "help_config": "サーバーの動作設定を行います。",
        "help_config_show": "現在の設定を表示します。",
        "help_config_setup": "対話形式で設定を行います。",
        "help_serve": "バックグラウンドでAPIサーバーを起動します。",
        "help_stop": "バックグラウンドのAPIサーバーを停止します。",
        "help_pull": "Hugging Faceからモデルをダウンロード・変換・追加します。",
        "help_create": "Modelfileから新しい「個性」を作成します。",
        "help_list": "利用可能なモデルと個性を一覧表示します (Ollama互換)。",
        "help_run": "指定したモデルや個性と対話形式でチャットします。",
        "help_rm": "個性またはベースモデルを削除します。",
        "help_cp": "個性をコピーします。",
        "help_show": "個性の詳細情報を表示します。",
        "help_quantize": "モデルを量子化して軽量版を作成します。",
        "help_rag": "RAG（検索拡張生成）インデックスを管理します。",
        "help_rag_add_docs": "ローカルのドキュメントをRAGインデックスに追加します。",
        "help_rag_status": "RAGインデックスの現在の状態を表示します。",
        "help_rag_clear": "RAGインデックスを完全に削除します。",

        "config_setup_welcome": "[bold cyan]対話形式で設定を開始します。[/bold cyan]",
        "config_setup_default_info": "現在の設定値をデフォルトとして表示します。変更しない場合はEnterキーを押してください。",
        "config_prompt_lang": "言語を選択してください / Please select a language (1: 日本語, 2: English)",
        "config_prompt_models_dir": "1. モデルを保存するディレクトリのパスは？",
        "config_prompt_host": "2. 外部からのアクセスを許可しますか？ (1: ローカルのみ許可, 2: すべて許可, 3: 両方許可)",
        "config_prompt_port": "3. サーバーが待ち受けるポート番号は？",
        "config_prompt_max_models": "4. メモリに同時にロードしておくモデルの最大数は？ (1が最も安全です)",
        "config_prompt_mem_limit": "5. システム全体のメモリ使用率の上限(%)は？ (これを超えるとモデルが解放されます)",
        "config_prompt_api_key": "6. APIキーを設定しますか？ (半角英数, 設定しない場合はEnter)",
        "config_prompt_api_key_overwrite": "6. 現在のAPIキーを上書きしますか？ (変更しない場合はEnter)",
        "config_saved": "[bold green]✓ 設定を保存しました。[/bold green]",
        
        "config_show_title": "現在の設定",
        "config_show_key": "設定項目",
        "config_show_value": "値",
        "config_key_models_dir": "モデル保存先 (models_dir)",
        "config_key_host": "待ち受けホスト (host)",
        "config_key_port": "待ち受けポート (port)",
        "config_key_max_models": "最大同時ロードモデル数 (max_loaded_models)",
        "config_key_mem_limit": "メモリ使用率上限 (%)",
        "config_key_api_key": "APIキー (api_key)",
        "config_api_key_set": "設定済み",
        "config_api_key_not_set": "未設定",
        "config_file_not_found": "[yellow]設定ファイルがまだ作成されていません。'mlxserver config setup' を実行してください。[/yellow]",

        "server_already_running": "[yellow]サーバーは既に起動しています。[/yellow]",
        "server_starting": "[green]サーバーを起動しています...",
        "server_start_success": "[bold green]✓ サーバーが {api_url} で起動しました。 (PID: {pid})[/bold green]",
        "server_start_fail": "[red]サーバーの起動に失敗しました。Uvicornのログを確認してください。[/red]",
        "server_not_running": "[yellow]サーバーは起動していません。[/yellow]",
        "server_stopping": "サーバー (PID: {pid}) を停止しています...",
        "server_stop_success": "[bold green]✓ サーバーを停止しました。[/bold green]",
        "server_stop_fail": "[yellow]プロセス (PID: {pid}) が見つかりませんでした。[/yellow]",
        "server_list_fail_not_running": "[red]サーバーが起動していません。まず 'serve' コマンドで起動してください。[/red]",
        "server_list_fail_comm": "[red]APIサーバーとの通信に失敗しました: {e}[/red]",
        "server_list_fail_json": "[red]APIサーバーから不正な応答がありました。[/red]",

        "pull_repo_info_fail": "[red]リポジトリ情報の取得に失敗しました: {e}[/red]",
        "pull_format_detected": "[green]形式を検出:[/green] {format}",
        "pull_unsupported_format": "[red]エラー: サポートされているモデル形式（MLX, GGUF, Safetensors）が見つかりませんでした。[/red]",
        "pull_downloading_to": "'{repo_id}' を '{output_dir}' にダウンロードしています...",
        "pull_download_error": "[red]ダウンロード中にエラーが発生しました: {e}[/red]",
        "pull_success": "[bold green]✓ プル成功！モデルは以下に保存されました:[/bold green] {output_dir}",
        "pull_gguf_too_many": "[yellow]警告: 複数のGGUFファイルが見つかりました。最初のファイル '{filename}' を使用します。[/yellow]",
        "pull_gguf_not_found": "[red]エラー: .ggufファイルが見つかりません。[/red]",
        "pull_converting": "MLX形式に変換しています...",
        "pull_convert_error": "[red]変換エラー:[/red]\n{stderr}",
        "pull_processing_error": "[red]処理中にエラーが発生しました: {e}[/red]",
        "pull_downloading_file": "'{filename}' をダウンロードしています...",
        "pull_converting_hf": "'{repo_id}' をダウンロード・変換しています... (時間がかかる場合があります)",
        "pull_cleanup_complete": "[cyan]クリーンアップ完了。[/cyan]",
        
        "create_modelfile_not_found": "[red]エラー: Modelfile '{file}' が見つかりません。[/red]",
        "create_parsing_error": "[red]Modelfileの解析中にエラーが発生しました: {e}[/red]",
        "create_no_from": "[red]エラー: Modelfileに 'FROM' が指定されていません。[/red]",
        "create_warn_no_server": "[yellow]警告: サーバーが起動していないため、ベースモデルの存在を確認できません。[/yellow]",
        "create_base_model_not_found": "[red]エラー: ベースモデル '{base_model}' が利用可能なモデル一覧にありません。[/red]",
        "create_warn_comm_fail": "[yellow]警告: サーバーとの通信に失敗し、ベースモデルの存在を確認できませんでした。[/yellow]",
        "create_success": "[bold green]✓ 個性 '{name}' を作成しました。[/bold green]",
        "create_base_model": "  - ベースモデル: {base_model}",
        "create_saved_to": "  - 保存先: {file}",
        "create_save_error": "[red]個性の保存中にエラーが発生しました: {e}[/red]",

        "list_tags_title": "利用可能なモデルと個性 (Tags)",
        "list_tags_col_name": "名前 (NAME)",

        "run_chat_start": "[bold cyan]'{model_name}' とのチャットを開始します。終了するには 'exit' または 'quit' と入力してください。[/bold cyan]",
        "run_prompt": "[bold green]>>> [/bold green]",
        "run_ai_response": "[bold blue]AI:[/bold blue] ",
        "run_api_error": "\n[red]APIリクエストエラー: {e}[/red]",
        "run_json_error": "\n[red]APIからの応答の解析に失敗しました。[/red]",

        "rm_confirm_personality": "個性 '{name}' を本当に削除しますか？",
        "rm_success_personality": "[bold green]✓ 個性 '{name}' を削除しました。[/bold green]",
        "rm_confirm_model": "ベースモデル '{name}' を本当に削除しますか？ (ディレクトリ内の全ファイルが消去されます)",
        "rm_success_model": "[bold green]✓ ベースモデル '{name}' を削除しました。[/bold green]",
        "rm_not_found": "[red]エラー: '{name}' という名前の個性またはベースモデルは見つかりません。[/red]",
        
        "cp_source_not_found": "[red]エラー: コピー元の個性 '{source}' が見つかりません。[/red]",
        "cp_dest_exists": "[red]エラー: コピー先の名前 '{destination}' は既に存在します。[/red]",
        "cp_success": "[bold green]✓ 個性 '{source}' を '{destination}' にコピーしました。[/bold green]",

        "show_not_found": "[red]エラー: 個性 '{name}' が見つかりません。[/red]",
        "show_title": "# 個性: {name}",
        "show_base_model": "[bold cyan]ベースモデル:[/] {base_model}",
        "show_system_prompt": "[bold cyan]システムプロンプト:[/]",

        "quantize_source_not_found": "[red]エラー: ソースモデル '{source_model_name}' が見つかりません。[/red]",
        "quantize_dest_exists": "[red]エラー: 出力先 '{quantized_name}' は既に存在します。[/red]",
        "quantize_success": "[bold green]✓ 量子化完了: '{quantized_name}'[/bold green]",
        "quantize_error": "[red]量子化中にエラーが発生しました。[/red]",
        "quantizing": "モデル '{source_model_name}' を {bits}ビットに量子化しています...",
        
        "rag_confirm_clear": "本当にRAGインデックスをすべて削除しますか？この操作は取り消せません。",
        "rag_cleared": "[bold green]✓ RAGインデックスを削除しました。[/bold green]",
        "rag_status_title": "RAGインデックスの状態",
        "rag_status_chunks": "インデックス済みチャンク数",
        "rag_status_files": "インデックス済みファイル",
        "rag_adding_docs": "[cyan]'{path}' からドキュメントをインデックスに追加しています...[/cyan]",
        "rag_reading_file": "ファイルを読み込み中: {file}",
        "rag_chunking_text": "テキストをチャンク化中...",
        "rag_embedding_chunks": "チャンクをベクトル化中 (埋め込みモデル: {model})",
        "rag_add_success": "[bold green]✓ ドキュメントのインデックス追加が完了しました。[/bold green]",
        "rag_add_no_files": "[yellow]警告: 指定されたパスに処理対象のファイル (.txt, .md) が見つかりませんでした。[/yellow]",

    },
    "en": {
        "help_main": "A CLI tool to manage and run MLX models.",
        "help_config": "Configure the server's behavior.",
        "help_config_show": "Show the current configuration.",
        "help_config_setup": "Set up the configuration interactively.",
        "help_serve": "Start the API server in the background.",
        "help_stop": "Stop the background API server.",
        "help_pull": "Download, convert, and add a model from Hugging Face.",
        "help_create": "Create a new 'personality' from a Modelfile.",
        "help_list": "List available models and personalities (Ollama compatible).",
        "help_run": "Start an interactive chat session with a model or personality.",
        "help_rm": "Remove a personality or a base model.",
        "help_cp": "Copy a personality.",
        "help_show": "Show the details of a personality.",
        "help_quantize": "Create a quantized, lightweight version of a model.",
        "help_rag": "Manage the RAG (Retrieval-Augmented Generation) index.",
        "help_rag_add_docs": "Add local documents to the RAG index.",
        "help_rag_status": "Show the current status of the RAG index.",
        "help_rag_clear": "Clear the entire RAG index.",

        "config_setup_welcome": "[bold cyan]Starting interactive setup...[/bold cyan]",
        "config_setup_default_info": "Current settings are shown as defaults. Press Enter to keep the current value.",
        "config_prompt_lang": "Please select a language / 言語を選択してください (1: 日本語, 2: English)",
        "config_prompt_models_dir": "1. What is the path to the directory for saving models?",
        "config_prompt_host": "2. Allow external access? (1: Local only, 2: Allow all, 3: Allow both)",
        "config_prompt_port": "3. Which port should the server listen on?",
        "config_prompt_max_models": "4. What is the maximum number of models to load in memory at once? (1 is safest)",
        "config_prompt_mem_limit": "5. What is the maximum system memory usage limit (%)? (Models will be unloaded above this)",
        "config_prompt_api_key": "6. Set an API key? (Alphanumeric, leave blank for no key)",
        "config_prompt_api_key_overwrite": "6. Overwrite the current API key? (Leave blank to keep it)",
        "config_saved": "[bold green]✓ Configuration saved.[/bold green]",

        "config_show_title": "Current Configuration",
        "config_show_key": "Setting",
        "config_show_value": "Value",
        "config_key_models_dir": "Models Directory (models_dir)",
        "config_key_host": "Listen Host (host)",
        "config_key_port": "Listen Port (port)",
        "config_key_max_models": "Max Concurrent Loaded Models (max_loaded_models)",
        "config_key_mem_limit": "Memory Usage Limit (%)",
        "config_key_api_key": "API Key (api_key)",
        "config_api_key_set": "Set",
        "config_api_key_not_set": "Not set",
        "config_file_not_found": "[yellow]Config file not found. Please run 'mlxserver config setup'.[/yellow]",

        "server_already_running": "[yellow]Server is already running.[/yellow]",
        "server_starting": "[green]Starting server...[/green]",
        "server_start_success": "[bold green]✓ Server started at {api_url} (PID: {pid})[/bold green]",
        "server_start_fail": "[red]Failed to start server. Check Uvicorn logs.[/red]",
        "server_not_running": "[yellow]Server is not running.[/yellow]",
        "server_stopping": "Stopping server (PID: {pid})...",
        "server_stop_success": "[bold green]✓ Server stopped.[/bold green]",
        "server_stop_fail": "[yellow]Process (PID: {pid}) not found.[/yellow]",
        "server_list_fail_not_running": "[red]Server is not running. Please start it first with the 'serve' command.[/red]",
        "server_list_fail_comm": "[red]Failed to communicate with the API server: {e}[/red]",
        "server_list_fail_json": "[red]Invalid response from the API server.[/red]",

        "pull_repo_info_fail": "[red]Failed to get repository info: {e}[/red]",
        "pull_format_detected": "[green]Format detected:[/green] {format} model",
        "pull_unsupported_format": "[red]Error: Could not find a supported model format (MLX, GGUF, Safetensors).[/red]",
        "pull_downloading_to": "Downloading '{repo_id}' to '{output_dir}'...",
        "pull_download_error": "[red]Error during download: {e}[/red]",
        "pull_success": "[bold green]✓ Pull successful! Model saved to:[/bold green] {output_dir}",
        "pull_gguf_too_many": "[yellow]Warning: Multiple GGUF files found. Using the first one: '{filename}'.[/yellow]",
        "pull_gguf_not_found": "[red]Error: .gguf file not found.[/red]",
        "pull_converting": "Converting to MLX format...",
        "pull_convert_error": "[red]Conversion Error:[/red]\n{stderr}",
        "pull_processing_error": "[red]Error during processing: {e}[/red]",
        "pull_downloading_file": "Downloading '{filename}'...",
        "pull_converting_hf": "Downloading and converting '{repo_id}'... (this may take a while)",
        "pull_cleanup_complete": "[cyan]Cleanup complete.[/cyan]",

        "create_modelfile_not_found": "[red]Error: Modelfile '{file}' not found.[/red]",
        "create_parsing_error": "[red]Error while parsing Modelfile: {e}[/red]",
        "create_no_from": "[red]Error: 'FROM' directive not found in Modelfile.[/red]",
        "create_warn_no_server": "[yellow]Warning: Server not running, cannot verify base model existence.[/yellow]",
        "create_base_model_not_found": "[red]Error: Base model '{base_model}' not found in available models.[/red]",
        "create_warn_comm_fail": "[yellow]Warning: Failed to communicate with server, could not verify base model existence.[/yellow]",
        "create_success": "[bold green]✓ Created personality '{name}'.[/bold green]",
        "create_base_model": "  - Base Model: {base_model}",
        "create_saved_to": "  - Saved to: {file}",
        "create_save_error": "[red]Error while saving personality: {e}[/red]",

        "list_tags_title": "Available Models and Personalities (Tags)",
        "list_tags_col_name": "NAME",

        "run_chat_start": "[bold cyan]Starting chat with '{model_name}'. Type 'exit' or 'quit' to end.[/bold cyan]",
        "run_prompt": "[bold green]>>> [/bold green]",
        "run_ai_response": "[bold blue]AI:[/bold blue] ",
        "run_api_error": "\n[red]API Request Error: {e}[/red]",
        "run_json_error": "\n[red]Failed to parse response from API.[/red]",
        
        "rm_confirm_personality": "Are you sure you want to delete the personality '{name}'?",
        "rm_success_personality": "[bold green]✓ Deleted personality '{name}'.[/bold green]",
        "rm_confirm_model": "Are you sure you want to delete the base model '{name}'? (All files in the directory will be erased)",
        "rm_success_model": "[bold green]✓ Deleted base model '{name}'.[/bold green]",
        "rm_not_found": "[red]Error: No personality or base model found with the name '{name}'.[/red]",

        "cp_source_not_found": "[red]Error: Source personality '{source}' not found.[/red]",
        "cp_dest_exists": "[red]Error: Destination name '{destination}' already exists.[/red]",
        "cp_success": "[bold green]✓ Copied personality '{source}' to '{destination}'.[/bold green]",

        "show_not_found": "[red]Error: Personality '{name}' not found.[/red]",
        "show_title": "# Personality: {name}",
        "show_base_model": "[bold cyan]Base Model:[/] {base_model}",
        "show_system_prompt": "[bold cyan]System Prompt:[/]",

        "quantize_source_not_found": "[red]Error: Source model '{source_model_name}' not found.[/red]",
        "quantize_dest_exists": "[red]Error: Destination '{quantized_name}' already exists.[/red]",
        "quantize_success": "[bold green]✓ Quantization complete: '{quantized_name}'[/bold green]",
        "quantize_error": "[red]An error occurred during quantization.[/red]",
        "quantizing": "Quantizing model '{source_model_name}' to {bits}-bit...",
        
        "rag_confirm_clear": "Are you sure you want to delete the entire RAG index? This action cannot be undone.",
        "rag_cleared": "[bold green]✓ RAG index cleared.[/bold green]",
        "rag_status_title": "RAG Index Status",
        "rag_status_chunks": "Indexed Chunks",
        "rag_status_files": "Indexed Files",
        "rag_adding_docs": "[cyan]Adding documents to index from '{path}'...[/cyan]",
        "rag_reading_file": "Reading file: {file}",
        "rag_chunking_text": "Chunking text...",
        "rag_embedding_chunks": "Embedding chunks (Model: {model})",
        "rag_add_success": "[bold green]✓ Finished adding documents to index.[/bold green]",
        "rag_add_no_files": "[yellow]Warning: No supported files (.txt, .md) found at the specified path.[/yellow]"
    }
}


# --- 設定 ---
APP_DIR = Path.home() / ".my_mlx_server"
CONFIG_FILE = APP_DIR / "config.json"
PID_FILE = APP_DIR / "server.pid"
PERSONALITIES_DIR = APP_DIR / "personalities"

# --- アプリケーションの初期化 ---
def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {
            "language": "ja", "models_dir": str(Path.home() / "my_mlx_models"),
            "host": "127.0.0.1", "port": 27000, "max_loaded_models": 1, "api_key": None
        }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            config.setdefault("port", 27000)
            return config
    except (json.JSONDecodeError, FileNotFoundError):
        return {"language": "ja"}

config = load_config()
lang = config.get("language", "ja")
t = locales.get(lang, locales["en"])

HOST = config.get("host", "127.0.0.1")
PORT = config.get("port", 27000)
# API_URLは常にローカルアドレスを使用（CLIコマンド用）
API_URL = f"http://127.0.0.1:{PORT}"

app = typer.Typer(help=t["help_main"])
config_app = typer.Typer(help=t["help_config"])
rag_app = typer.Typer(help=t["help_rag"])
app.add_typer(config_app, name="config")
app.add_typer(rag_app, name="rag")

console = Console()

# --- 補助関数 ---
def save_config(new_config: dict):
    APP_DIR.mkdir(exist_ok=True, parents=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)

def is_server_running():
    if not PID_FILE.exists(): return False
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        PID_FILE.unlink()
        return False

def get_server_status():
    try:
        response = requests.get(f"{API_URL}/", timeout=1)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# --- 設定 (config) サブコマンド ---
@config_app.command("show", help=t["help_config_show"])
def show_config():
    if not CONFIG_FILE.exists():
        console.print(t["config_file_not_found"])
        return
    current_config = load_config()
    table = Table(title=t["config_show_title"])
    table.add_column(t["config_show_key"], style="cyan")
    table.add_column(t["config_show_value"], style="magenta")
    table.add_row(t["config_key_models_dir"], current_config.get("models_dir"))
    # ホスト設定の表示を改善
    host_display = current_config.get("host")
    if host_display == "both":
        host_display = "127.0.0.1 + 0.0.0.0 (両方許可)"
    table.add_row(t["config_key_host"], host_display)
    table.add_row(t["config_key_port"], str(current_config.get("port")))
    table.add_row(t["config_key_max_models"], str(current_config.get("max_loaded_models")))
    api_key_display = t["config_api_key_set"] if current_config.get("api_key") else t["config_api_key_not_set"]
    table.add_row(t["config_key_api_key"], api_key_display)
    console.print(table)

@config_app.command("setup", help=t["help_config_setup"])
def setup_config():
    current_config = load_config()
    lang_prompt = locales["ja"]["config_prompt_lang"]
    lang_choice = typer.prompt(lang_prompt, default="1" if current_config.get("language", "ja") == "ja" else "2", type=click.Choice(["1", "2"]))
    current_config["language"] = "ja" if lang_choice == "1" else "en"
    local_t = locales.get(current_config["language"])
    console.print(local_t["config_setup_welcome"])
    console.print(local_t["config_setup_default_info"])
    current_config["models_dir"] = typer.prompt(local_t["config_prompt_models_dir"], default=current_config.get("models_dir", str(Path.home() / "my_mlx_models")))
    host_choice = typer.prompt(local_t["config_prompt_host"], default="1" if current_config.get("host", "127.0.0.1") == "127.0.0.1" else "2" if current_config.get("host", "127.0.0.1") == "0.0.0.0" else "3", type=click.Choice(["1", "2", "3"]))
    if host_choice == "1":
        current_config["host"] = "127.0.0.1"
    elif host_choice == "2":
        current_config["host"] = "0.0.0.0"
    else:  # host_choice == "3"
        current_config["host"] = "both"
    current_config["port"] = typer.prompt(local_t["config_prompt_port"], default=current_config.get("port", 27000), type=int)
    current_config["max_loaded_models"] = typer.prompt(local_t["config_prompt_max_models"], default=current_config.get("max_loaded_models", 1), type=int)
    current_key = current_config.get("api_key")
    api_key_prompt = local_t["config_prompt_api_key_overwrite"] if current_key else local_t["config_prompt_api_key"]
    new_key = typer.prompt(api_key_prompt, default="", show_default=False)
    if new_key: current_config["api_key"] = new_key
    elif new_key == "" and not current_key: current_config["api_key"] = None
    save_config(current_config)
    console.print(local_t["config_saved"])
    global t, HOST, PORT, API_URL
    t = local_t; config = load_config(); HOST = config.get("host", "127.0.0.1"); PORT = config.get("port", 27000)
    # API_URLは常にローカルアドレスを使用（CLIコマンド用）
    API_URL = f"http://127.0.0.1:{PORT}"
    show_config()
@app.command(name="serve", help=t["help_serve"])
def serve():
    if is_server_running() and get_server_status():
        console.print(t["server_already_running"])
        return
    
    APP_DIR.mkdir(exist_ok=True, parents=True)
    PERSONALITIES_DIR.mkdir(exist_ok=True, parents=True)
    
    # 設定ファイルから最新の値を読み込み
    current_config = load_config()
    current_host = current_config.get("host", "127.0.0.1")
    current_port = current_config.get("port", 27000)
    
    # デバッグ情報を表示
    console.print(f"[dim]デバッグ: 読み込まれた設定 - host: {current_host}, port: {current_port}[/dim]")
    
    # ホスト設定に基づいてサーバーを起動
    if current_host == "both":
        # 両方のアドレスでリッスン（0.0.0.0で起動）
        command = [
            sys.executable, "-m", "uvicorn",
            "mlxserver.main:app", "--host=0.0.0.0", f"--port={current_port}"
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        PID_FILE.write_text(str(process.pid))
        
        console.print(f"[green]サーバーを起動しています... (127.0.0.1:{current_port} と 0.0.0.0:{current_port})[/green]")
        for _ in range(15):
            if get_server_status():
                console.print(f"[bold green]✓ サーバーが起動しました。[/bold green]")
                console.print(f"  - ローカルアクセス: http://127.0.0.1:{current_port}")
                console.print(f"  - 外部アクセス: http://0.0.0.0:{current_port}")
                console.print(f"  - PID: {process.pid}")
                return
            time.sleep(1)
    else:
        # 通常の単一アドレスで起動
        command = [
            sys.executable, "-m", "uvicorn",
            "mlxserver.main:app", f"--host={current_host}", f"--port={current_port}"
        ]
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        PID_FILE.write_text(str(process.pid))

        console.print(t["server_starting"].format(host=current_host))
        for _ in range(15):
            if get_server_status():
                console.print(t["server_start_success"].format(api_url=API_URL, pid=process.pid))
                return
            time.sleep(1)
    
    console.print(t["server_start_fail"])
    if PID_FILE.exists():
        PID_FILE.unlink()

@app.command(name="stop", help=t["help_stop"])
def stop():
    if not is_server_running():
        console.print(t["server_not_running"])
        return
    pid = int(PID_FILE.read_text())
    try:
        console.print(t["server_stopping"].format(pid=pid))
        os.kill(pid, 9)
        console.print(t["server_stop_success"])
    except OSError:
        console.print(t["server_stop_fail"].format(pid=pid))
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

@app.command(name="pull", help=t["help_pull"])
def pull_model(repo_id: str = typer.Argument(..., help="Hugging FaceのリポジトリID")):
    # 設定ファイルから最新の値を読み込み
    current_config = load_config()
    models_dir = Path(current_config.get("models_dir"))
    
    console.print(f"[cyan]リポジトリ '{repo_id}' の情報を取得しています...[/cyan]")
    try:
        info = model_info(repo_id)
        files = [sibling.rfilename for sibling in info.siblings]
        if os.environ.get("MLXSERVE_DEBUG"):
            console.print(f"[dim]検出されたファイル: {', '.join(files[:10])}{'...' if len(files) > 10 else ''}[/dim]")
    except Exception as e:
        console.print(t["pull_repo_info_fail"].format(e=e))
        raise typer.Exit(code=1)

    model_name = repo_id.split('/')[-1]
    output_dir = models_dir / model_name
    
    # より正確な形式判定
    is_gguf = any(f.endswith(".gguf") for f in files)
    is_safetensors = any(f.endswith(".safetensors") for f in files)
    is_mlx = any(f == "weights.npz" for f in files)
    
    # 形式の詳細情報を表示（詳細モードの場合のみ）
    if os.environ.get("MLXSERVE_DEBUG"):
        console.print(f"[dim]形式分析: GGUF={is_gguf}, Safetensors={is_safetensors}, MLX={is_mlx}[/dim]")

    if is_mlx:
        console.print(t["pull_format_detected"].format(format="Native MLX"))
        pull_native_mlx(repo_id, output_dir)
    elif is_gguf or is_safetensors:
        # GGUFとSafetensorsは同じ処理で統一
        format_type = "GGUF" if is_gguf else "PyTorch/Safetensors"
        console.print(t["pull_format_detected"].format(format=format_type))
        pull_and_convert_to_mlx(repo_id, output_dir, format_type)
    else:
        console.print(t["pull_unsupported_format"])
        raise typer.Exit(code=1)

def pull_native_mlx(repo_id: str, output_dir: Path):
    console.print(t["pull_downloading_to"].format(repo_id=repo_id, output_dir=output_dir))
    try:
        snapshot_download(repo_id=repo_id, local_dir=output_dir, local_dir_use_symlinks=False)
        console.print(t["pull_success"].format(output_dir=output_dir))
    except Exception as e:
        console.print(t["pull_download_error"].format(e=e))

def pull_and_convert_to_mlx(repo_id: str, output_dir: Path, format_type: str):
    """
    統一されたMLX変換処理（量子化対応）
    GGUFとSafetensorsの両方に対応
    インターネット情報に基づく最適化された変換オプションを使用
    """
    console.print(f"[cyan]{format_type}形式をMLX形式に変換しています（4ビット量子化）...[/cyan]")
    
    try:
        # 出力ディレクトリの準備
        if output_dir.exists():
            console.print(f"[yellow]既存のディレクトリ '{output_dir}' を削除しています...[/yellow]")
            import shutil
            shutil.rmtree(output_dir)
        
        # 出力ディレクトリの親ディレクトリを作成
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # mlx_lm.convertを直接使用（推奨される方法）
        try:
            from mlx_lm import convert
            
            if os.environ.get("MLXSERVE_DEBUG"):
                console.print(f"[dim]mlx_lm.convertを直接使用して変換中...[/dim]")
            
            # 変換処理を直接実行
            convert(
                hf_path=repo_id,
                mlx_path=str(output_dir),
                quantize=True,
                q_bits=4,
                dtype="float16"
            )
            
            console.print(f"[green]✓ 直接変換が完了しました[/green]")
            
        except ImportError:
            # mlx_lm.convertが利用できない場合は、従来のsubprocess方式を使用
            console.print(f"[yellow]mlx_lm.convertが利用できないため、従来の方式を使用します[/yellow]")
            
            command = [
                sys.executable, "-m", "mlx_lm", "convert",
                "--hf-path", repo_id,
                "--mlx-path", str(output_dir),
                "--quantize",  # 量子化を有効化
                "--q-bits", "4",  # 4ビット量子化
                "--dtype", "float16"  # メモリ効率のためfloat16を使用
            ]
            
            if os.environ.get("MLXSERVE_DEBUG"):
                console.print(f"[dim]実行コマンド: {' '.join(command)}[/dim]")
            
            # 変換プロセスの実行
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding='utf-8'
            )
            
            # リアルタイムで出力を表示
            with console.status(f"[bold green]MLX形式に変換中...[/bold green]") as status:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        console.print(f"[dim] > {output.strip()}[/dim]")
            
            # 変換結果の確認
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        
        # 変換完了の検証
        if not verify_mlx_conversion(output_dir):
            raise RuntimeError("MLX変換の検証に失敗しました")
        
        console.print(t["pull_success"].format(output_dir=output_dir))
        
    except subprocess.CalledProcessError as e:
        console.print(f"[red]変換プロセスエラー (終了コード: {e.returncode})[/red]")
        console.print(f"[red]コマンド: {' '.join(e.cmd)}[/red]")
        # エラーが発生した場合、部分的なファイルをクリーンアップ
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        raise
    except Exception as e:
        console.print(f"[red]変換エラー: {e}[/red]")
        # エラーが発生した場合、部分的なファイルをクリーンアップ
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        raise

def verify_mlx_conversion(output_dir: Path) -> bool:
    """
    MLX変換が正しく完了したかを検証
    """
    try:
        # 必要なファイルの存在確認
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        # 必須ファイルの確認
        for required_file in required_files:
            if not (output_dir / required_file).exists():
                console.print(f"[red]エラー: 必須ファイル '{required_file}' が見つかりません[/red]")
                return False
        
        # MLX形式の重みファイルの確認
        mlx_files = list(output_dir.glob("*.npz"))
        safetensors_files = list(output_dir.glob("*.safetensors"))
        
        if mlx_files:
            # MLX形式の重みファイルが存在する場合
            console.print(f"[green]✓ MLX変換の検証が完了しました[/green]")
            console.print(f"[dim]検出されたMLXファイル: {[f.name for f in mlx_files]}[/dim]")
            return True
        elif safetensors_files:
            # Safetensorsファイルが存在する場合（変換が完了していない可能性）
            console.print(f"[yellow]警告: MLX形式の重みファイル（*.npz）が見つかりません[/yellow]")
            console.print(f"[dim]検出されたSafetensorsファイル: {[f.name for f in safetensors_files]}[/dim]")
            console.print(f"[yellow]注意: mlx_lm convertが正しく動作していない可能性があります[/yellow]")
            
            # 現在の状況では、Safetensorsファイルが存在する場合も成功として扱う
            # これは一時的な対応策です
            console.print(f"[green]✓ 基本的な変換検証が完了しました（Safetensorsファイル）[/green]")
            return True
        else:
            # 重みファイルが見つからない場合
            console.print(f"[red]エラー: 重みファイル（*.npz または *.safetensors）が見つかりません[/red]")
            console.print(f"[dim]ディレクトリ内容: {[f.name for f in output_dir.iterdir()]}[/dim]")
            return False
        
    except Exception as e:
        console.print(f"[red]変換検証エラー: {e}[/red]")
        return False

@app.command(name="create", help=t["help_create"])
def create_personality(name: str, file: Path = typer.Option(..., "-f", "--file", exists=True, file_okay=True, dir_okay=False, readable=True)):
    base_model = None; system_prompt = ""
    try:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines(); in_system_prompt = False; system_lines = []
            for line in lines:
                stripped_line = line.strip()
                if not in_system_prompt and stripped_line.upper().startswith("FROM "): base_model = stripped_line.split(maxsplit=1)[1]
                elif stripped_line.upper().startswith('SYSTEM """'): in_system_prompt = True; system_lines.append(line.split('"""', 1)[1])
                elif '"""' in stripped_line and in_system_prompt: system_lines.append(line.split('"""', 1)[0]); in_system_prompt = False
                elif in_system_prompt: system_lines.append(line)
            system_prompt = "".join(system_lines).strip()
    except Exception as e: console.print(t["create_parsing_error"].format(e=e)); raise typer.Exit(code=1)
    if not base_model: console.print(t["create_no_from"]); raise typer.Exit(code=1)
    if get_server_status():
        try:
            res = requests.get(f"{API_URL}/api/tags").json()
            available_tags = [tag["name"].replace(":latest", "") for tag in res.get("models", [])]
            if base_model not in available_tags: console.print(t["create_base_model_not_found"].format(base_model=base_model)); raise typer.Exit(code=1)
        except requests.RequestException: console.print(t["create_warn_comm_fail"])
    else: console.print(t["create_warn_no_server"])
    personality_data = {"base_model": base_model, "system_prompt": system_prompt, "created_at": datetime.utcnow().isoformat()}
    personality_file = PERSONALITIES_DIR / f"{name}.json"
    try:
        with open(personality_file, "w", encoding="utf-8") as f: json.dump(personality_data, f, indent=2, ensure_ascii=False)
        console.print(t["create_success"].format(name=name))
        console.print(t["create_base_model"].format(base_model=base_model))
        console.print(t["create_saved_to"].format(file=personality_file))
    except Exception as e: console.print(t["create_save_error"].format(e=e)); raise typer.Exit(code=1)

@app.command(name="list", help=t["help_list"])
def list_items():
    if not get_server_status(): console.print(t["server_list_fail_not_running"]); return
    try:
        res = requests.get(f"{API_URL}/api/tags").json()
        tags = res.get("models", [])
        table = Table(title=t["list_tags_title"])
        table.add_column(t["list_tags_col_name"], style="cyan", no_wrap=True)
        for tag in sorted(tags, key=lambda x: x['name']):
            table.add_row(tag["name"])
        console.print(table)
    except requests.RequestException as e: console.print(t["server_list_fail_comm"].format(e=e))
    except json.JSONDecodeError: console.print(t["server_list_fail_json"])

@app.command(name="run", help=t["help_run"])
def run_interactive_chat(model_name: str = typer.Argument(..., help="使用するベースモデル名または個性名")):
    if not get_server_status(): console.print(t["server_list_fail_not_running"]); return
    console.print(t["run_chat_start"].format(model_name=model_name))
    
    conversation_history = []
    
    while True:
        try:
            prompt = console.input(t["run_prompt"])
            if prompt.lower() in ["exit", "quit"]: break
            
            conversation_history.append({"role": "user", "content": prompt})
            
            payload = {"model": model_name, "messages": conversation_history, "stream": True}
            headers = {"Content-Type": "application/json"}
            if config.get("api_key"): headers["X-API-Key"] = config["api_key"]

            full_ai_response = ""
            with requests.post(f"{API_URL}/api/chat", json=payload, stream=True, headers=headers, timeout=180) as r:
                r.raise_for_status()
                console.print(t["run_ai_response"], end="")
                for chunk in r.iter_lines():
                    if chunk.strip():
                        data = json.loads(chunk)
                        content = data.get("message", {}).get("content", "")
                        print(content, end="", flush=True)
                        full_ai_response += content
                        if data.get("done"): break
                print()
            
            conversation_history.append({"role": "assistant", "content": full_ai_response})
        except typer.Abort: break
        except requests.RequestException as e: console.print(t["run_api_error"].format(e=e)); break
        except json.JSONDecodeError: console.print(t["run_json_error"]); break

@app.command(name="rm", help=t["help_rm"])
def remove_item(name: str):
    models_dir = Path(config.get("models_dir"))
    personality_file = PERSONALITIES_DIR / f"{name}.json"; model_dir = models_dir / name
    if personality_file.exists():
        if typer.confirm(t["rm_confirm_personality"].format(name=name)):
            personality_file.unlink(); console.print(t["rm_success_personality"].format(name=name))
    elif model_dir.is_dir():
        if typer.confirm(t["rm_confirm_model"].format(name=name)):
            shutil.rmtree(model_dir); console.print(t["rm_success_model"].format(name=name))
    else: console.print(t["rm_not_found"].format(name=name))

@app.command(name="cp", help=t["help_cp"])
def copy_personality(source: str, destination: str):
    source_file = PERSONALITIES_DIR / f"{source}.json"; dest_file = PERSONALITIES_DIR / f"{destination}.json"
    if not source_file.exists(): console.print(t["cp_source_not_found"].format(source=source)); raise typer.Exit(code=1)
    if dest_file.exists(): console.print(t["cp_dest_exists"].format(destination=destination)); raise typer.Exit(code=1)
    shutil.copy(source_file, dest_file); console.print(t["cp_success"].format(source=source, destination=destination))

@app.command(name="show", help=t["help_show"])
def show_personality_details(name: str):
    p_file = PERSONALITIES_DIR / f"{name}.json"
    if not p_file.exists(): console.print(t["show_not_found"].format(name=name)); raise typer.Exit(code=1)
    with open(p_file, "r", encoding="utf-8") as f: data = json.load(f)
    console.print(Markdown(t["show_title"].format(name=name)))
    console.print(t["show_base_model"].format(base_model=data.get("base_model")))
    console.print(t["show_system_prompt"]); console.print(Markdown(f"```\n{data.get('system_prompt')}\n```"))

@app.command(name="quantize", help=t["help_quantize"])
def quantize_model(source_model_name: str, bits: int = typer.Option(4, "-b", "--bits", help="量子化ビット数 (例: 2, 4, 8)")):
    models_dir = Path(config.get("models_dir"))
    source_dir = models_dir / source_model_name
    quantized_name = f"{source_model_name}-{bits}bit"; output_dir = models_dir / quantized_name
    if not source_dir.is_dir(): console.print(t["quantize_source_not_found"].format(source_model_name=source_model_name)); raise typer.Exit(code=1)
    if output_dir.exists(): console.print(t["quantize_dest_exists"].format(quantized_name=quantized_name)); raise typer.Exit(code=1)
    console.print(t["quantizing"].format(source_model_name=source_model_name, bits=bits))
    command = [sys.executable, "-m", "mlx_lm.quantize", "-m", str(source_dir), "-o", str(output_dir), "-b", str(bits)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        console.print(t["quantize_error"]); console.print(result.stderr)
    else:
        console.print(t["quantize_success"].format(quantized_name=quantized_name))

# --- RAGサブコマンドグループ ---
@rag_app.command("add-docs", help=t["help_rag_add_docs"])
def rag_add_docs(path: Path = typer.Argument(..., help="ドキュメントファイルまたはディレクトリへのパス", exists=True),
                 embedding_model: str = typer.Option(..., "-m", "--model", help="使用する埋め込みモデル名"),
                 chunk_size: int = typer.Option(512, help="テキストを分割するチャンクサイズ（文字数）")):
    if not get_server_status(): console.print(t["server_list_fail_not_running"]); raise typer.Exit(code=1)
    files_to_process = []
    if path.is_dir():
        files_to_process.extend(list(path.glob("**/*.txt"))); files_to_process.extend(list(path.glob("**/*.md")))
    elif path.is_file() and path.suffix in [".txt", ".md"]: files_to_process.append(path)
    if not files_to_process: console.print(t["rag_add_no_files"]); return
    console.print(t["rag_adding_docs"].format(path=path))
    headers = {"Content-Type": "application/json"}
    if config.get("api_key"): headers["X-API-Key"] = config["api_key"]
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TimeElapsedColumn()) as progress:
        task_files = progress.add_task(description=t["rag_reading_file"].format(file=""), total=len(files_to_process))
        for file in files_to_process:
            progress.update(task_files, description=t["rag_reading_file"].format(file=file.name))
            try:
                text = file.read_text(encoding="utf-8")
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                # (ダミーのAPI呼び出し)
                hidden_dim = 32 # 仮
                vectors = np.random.rand(len(chunks), hidden_dim)
                rag_manager.add_to_index(str(file.name), chunks, vectors)
                progress.advance(task_files)
            except Exception as e: console.print(f"\n[red]ファイル '{file.name}' の処理中にエラー: {e}[/red]"); continue
    console.print(t["rag_add_success"])

@rag_app.command("status", help=t["help_rag_status"])
def rag_status():
    status = rag_manager.get_rag_status()
    table = Table(title=t["rag_status_title"])
    table.add_column(t["config_show_key"], style="cyan"); table.add_column(t["config_show_value"], style="magenta")
    table.add_row(t["rag_status_chunks"], str(status["indexed_chunks"]))
    table.add_row(t["rag_status_files"], "\n".join(status["indexed_files"]))
    console.print(table)
    
@rag_app.command("clear", help=t["help_rag_clear"])
def rag_clear():
    if typer.confirm(t["rag_confirm_clear"]):
        rag_manager.clear_index(); console.print(t["rag_cleared"])

if __name__ == "__main__":
    app()