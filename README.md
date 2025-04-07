# 🧠 マルチエージェントAIシステム - 実行手順（Mac向け）

このアプリは、OpenAI GPT-4を利用して複数のAIエージェントを自動で生成し、指示に基づいたタスクの分解、非同期処理、レビュー、差し戻しを繰り返しながら成果物を生成するシステムです。
(GPT-4だとコストが高すぎたので、GPT-4miniにしている)

## ✅ 必要要件
- macOS (最新版推奨)
- RustとXCodeが入っている
- Python 3.10〜3.12
- OpenAIのAPIキー

## 📦 セットアップ手順

### 0. 環境の確認
```bash
rustc --version
xcode-select -p
python --version ## >=3.10, <=3.12
```

### 1. Pythonの仮想環境を作成
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. 必要なパッケージをインストール
```bash
pip install tiktoken streamlit openai crewai python-dotenv watchdog langchain-openai faker
```

### 3. `.env` ファイルを作成
ルートディレクトリに `.env` ファイルを作成し、以下のように記述してください：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

※ `sk-...` はご自身のOpenAI APIキーに置き換えてください。

### 4. アプリの起動
```bash
streamlit run app.py
```

初回実行時にブラウザが開き、GUIが表示されます。

## 🧪 使い方
1. テキストエリアに「作ってほしい資料」や「調査してほしいこと」など、自由な依頼を入力します。
2. 「実行」ボタンを押すと、エージェントが自動で編成され、作業・レビュー・差し戻しを繰り返して最終成果を表示します。

## 🛠 オプション（任意）
### エラー対処
- `.env` ファイルが存在しない場合、API接続に失敗します。
- OpenAIのAPI使用上限に注意してください。

### その他の依存解消
```bash
pip install --upgrade pip
```

### ① Xcode Command Line Tools をインストール（または再インストール）
```bash
xcode-select --install
```
すでに入っている場合でも、トラブル防止のために以下で再設定するとよいです

```bash
sudo xcode-select --reset
```

### ② rust をインストール（tiktoken はRustを使ってます）
```bash
brew install rust
```

③ tiktoken のインストールを再試行
Rustとビルド環境が整った状態で：
```bash
pip install tiktoken
```

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt  # または必要なパッケージをまとめて

## 📁 フォルダ構成（例）
```
project-folder/
├── app.py
├── .env
└── README.md
```

## 🚀 今後の拡張候補
- 中間出力のGUI表示
- PDFエクスポート
- Slack通知との連携
- 成果物の品質評価スコア表示

---

本アプリは教育・研究・プロトタイピング用途を想定しています。商用利用の際はAPI利用料等にご注意ください。

