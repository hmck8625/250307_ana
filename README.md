# requirements.txt
streamlit==1.26.0
pandas==2.0.3
numpy==1.24.3
plotly==5.15.0
gspread==5.10.0
oauth2client==4.1.3
google-auth==2.22.0
openai==0.28.0
python-dotenv==1.0.0

# .streamlit/config.toml
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

# .streamlit/secrets.toml
# 以下は例です。実際のデプロイ時には適切な認証情報に置き換えてください
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "your-private-key"
client_email = "your-client-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"

[openai]
api_key = "your-openai-api-key"

# README.md
# 広告パフォーマンス分析システム

このアプリケーションは、広告パフォーマンスデータを分析し、洞察と推奨事項を提供するStreamlitベースのツールです。

## 主な機能
- Google Spreadsheetからのデータ読み込み
- 期間比較分析（月次/週次/カスタム）
- CV増減の寄与度分析
- CPA変化要因分析
- 媒体グループ・パターン分析
- ChatGPT APIを使用した分析レポート生成

## セットアップ方法
1. このリポジトリをクローンします。
2. 必要なライブラリをインストールします: `pip install -r requirements.txt`
3. `.streamlit/secrets.toml` ファイルを作成し、Google APIとOpenAI APIの認証情報を設定します。
4. アプリケーションを実行します: `streamlit run app.py`

## 認証設定方法
1. Google APIコンソールでサービスアカウントを作成し、JSONキーをダウンロードします。
2. `.streamlit/secrets.toml` ファイルに認証情報を追加します。
3. OpenAI APIキーを取得し、`.streamlit/secrets.toml` ファイルに追加します。

## 使用例
1. データソース設定でGoogle SpreadsheetのURLとシート名を入力します。
2. 期間比較分析タブで分析したい期間と粒度を選択します。
3. 分析を実行すると、結果が表示されます。
4. レポート出力タブでChatGPTが生成した分析レポートを確認できます。

## デプロイ方法
1. Streamlit Cloudにアカウントを作成します。
2. このリポジトリを連携します。
3. シークレット情報を設定します。
4. デプロイを実行します。

## 注意事項
- API認証情報は厳重に管理してください。
- 大量のデータを扱う場合はパフォーマンスに注意してください。
- OpenAI APIの使用にはコストがかかる場合があります。

詳細な情報やサポートについては、[プロジェクトドキュメント](link-to-documentation)を参照してください。
