mlxserver: A Powerful, Local AI Model Server for macOS
mlxserverは、Apple Silicon (Mシリーズチップ) に最適化されたローカルAIモデルサーバーです。Ollamaにインスパイアされ、MLXフレームワークのパワーを最大限に引き出しながら、より高いカスタマイズ性と拡張性を提供することを目指しています。

対話的なCLI、自動的なモデル管理、RAG（検索拡張生成）、オンザフライでのモデル変換など、ローカルAI開発を加速するための豊富な機能を備えています。

✨ 主な機能
🧠 高度なモデル管理:

スマートなpullコマンド: Hugging Face上のモデル形式（ネイティブMLX, GGUF, Safetensors）を自動判別し、変換から配置までをワンストップで行います。

個性（Personalities）: Modelfileを使って、ベースモデルに特定の役割やシステムプロンプトを与えた「個性」を無制限に作成できます (create)。

オンザフライ量子化: フルサイズのモデルを、コマンド一つで軽量な2/4/8ビット版に変換できます (quantize)。

🤖 対話的なインターフェース:

CLIチャット: ターミナル上で直接AIと対話できます (run)。

ストリーミング応答: ChatGPTのように、AIの応答がリアルタイムで一文字ずつ表示されます。

設定ウィザード: 対話形式でサーバーの全設定を簡単に行えます (config setup)。

🚀 パワフルなAPIサーバー:

自動モデル管理: APIリクエストに応じてモデルを自動でメモリにロードし、一定時間使われないと自動でアンロードするため、リソースを効率的に利用できます。

LRUキャッシュ: メモリ上限に達した際に、最も使われていないモデルを自動で解放します。

RAG（検索拡張生成）: 手持ちのドキュメントをインデックス化し、その内容に基づいてAIに応答させることが可能です。

埋め込みAPI: テキストをベクトル化するAPIエンドポイント (/embeddings) を提供します。

🔧 高い柔軟性:

国際化対応: CLIの表示言語を日本語と英語で切り替え可能です。

セキュリティ: APIキーによるアクセス制限を設定できます。

ネットワーク設定: 待ち受けホストやポートを自由に設定できます。

⚙️ インストール
前提条件の準備

Homebrewをインストールします。

pyenvをインストールして、最新のPython（3.11以上推奨）を導入します。

brew install pyenv
pyenv install 3.11.9

リポジトリのクローン

git clone <YOUR_REPOSITORY_URL>
cd mlxserve

Python環境の設定
プロジェクトディレクトリで、使用するPythonのバージョンを指定します。

pyenv local 3.11.9

仮想環境の作成と有効化

python -m venv .venv
source .venv/bin/activate

インストール
プロジェクトの依存関係をインストールし、mlxserverコマンドを有効化します。

python -m pip install --upgrade pip setuptools
python -m pip install -e .

🛠️ 初期設定
インストール後、最初に以下のコマンドを実行して、対話形式でサーバーの基本設定を行ってください。

mlxserver config setup

言語、モデルの保存先、待ち受けポート、APIキーなどを設定できます。設定内容は ~/.my_mlx_server/config.json に保存されます。

🚀 使い方 (コマンドリファレンス)
サーバー管理

サーバーを起動する:

mlxserver serve

サーバーを停止する:

mlxserver stop

モデルと個性の管理

モデルを追加する (Pull):
Hugging Faceからモデルをダウンロード・変換・追加します。形式は自動で判別されます。

# ネイティブMLXモデル
mlxserver pull apple/mlx-community/Phi-3-mini-4k-instruct-8bit

# GGUFモデル
mlxserver pull mlx-community/Mistral-7B-Instruct-v0.2-GGUF

# PyTorch/Safetensorsモデル
mlxserver pull meta-llama/Meta-Llama-3-8B-Instruct

一覧を表示する:
ローカルにあるベースモデル、ロード中のモデル、作成済みの個性を一覧表示します。

mlxserver list

個性を作成する (Create):
まず、Modelfileという名前のテキストファイルを作成します。

# ベースとなるモデル
FROM Meta-Llama-3-8B-Instruct

# AIの役割や口調を定義
SYSTEM """
あなたは猫のキャラクター「ニャンコ先生」です。
語尾には必ず「〜ニャ」を付けて、可愛らしく親しみやすい口調で話してください。
"""

次に、createコマンドで「個性」として登録します。

mlxserver create nyan-sensei -f ./Modelfile

詳細を表示する (Show):

mlxserver show nyan-sensei

コピーする (Cp):

mlxserver cp nyan-sensei nyan-sensei-v2

削除する (Rm):

mlxserver rm nyan-sensei
mlxserver rm Meta-Llama-3-8B-Instruct

量子化する (Quantize):
モデルを軽量な4ビット版に変換します。

mlxserver quantize Meta-Llama-3-8B-Instruct --bits 4

対話とRAG

対話を開始する (Run):
作成した個性とターミナルでチャットします。

mlxserver run nyan-sensei

RAGインデックスを管理する:

ドキュメントを追加: 手持ちのテキストファイルをAIが参照できるようにします。

# RAGには埋め込みモデルが必要。まずpullしておく。
mlxserver pull BAAI/bge-small-en-v1.5 

# ドキュメントをインデックスに追加
mlxserver rag add-docs ./my-notes/ --model BAAI/bge-small-en-v1.5

状態を確認: mlxserver rag status

インデックスを削除: mlxserver rag clear

RAGを使って対話する:
--ragフラグと、インデックス作成時に使った埋め込みモデルを指定します。

mlxserver run nyan-sensei --rag --embedding-model BAAI/bge-small-en-v1.5

🔌 APIの使い方
サーバーを起動すると、http://<host>:<port>でAPIが利用可能になります。

エンドポイント: /generate
メソッド: POST
ヘッダー:

Content-Type: application/json

X-API-Key: <設定したAPIキー> (APIキーを設定した場合)

リクエストボディ (例):

{
  "personality_name": "nyan-sensei",
  "prompt": "今日の調子はどう？",
  "max_tokens": 50,
  "stream": true,
  "use_rag": false
}

curlでの実行例:

curl -N -X POST [http://127.0.0.1:27000/generate](http://127.0.0.1:27000/generate) \
-H "Content-Type: application/json" \
-H "X-API-Key: your-secret-key" \
-d '{
  "personality_name": "nyan-sensei",
  "prompt": "自己紹介して！",
  "stream": true
}'

© ライセンス
このプロジェクトはMITライセンスの下で公開されています。