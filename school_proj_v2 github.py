import os
# --- 修正 Render 上 ChromaDB 的 SQLite 版本問題 (必須放在最上面) ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ----------------------------------------------------------------

from flask import Flask, request, abort
from openai import OpenAI
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError, LineBotApiError
import traceback

# 用 Chroma 和 Google Embeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

app = Flask(__name__)

# --- 🔒 安全性修改：從環境變數讀取 Key，不要寫在程式碼裡 ---
# 如果讀不到環境變數，程式會報錯提醒你，這樣比較安全
AI_API_KEY = os.environ.get("GEMINI_API_KEY")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")

# 檢查 Key 是否存在
if not all([AI_API_KEY, LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET]):
    print(" 錯誤：請在 Render 的 Environment Variables 設定 API Key！")

# 初始化 Client
client = OpenAI(
    api_key=AI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = "gemini-3-flash-preview"

# 初始化 Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", 
    google_api_key=AI_API_KEY,
    task_type="retrieval_query"
)

# 載入 Chroma 資料庫
DB_PATH = "my_vector_db"
vectorstore = None

try:
    if os.path.exists(DB_PATH):
        vectorstore = Chroma(
            persist_directory=DB_PATH, 
            embedding_function=embeddings
        )
        print(f"Chroma 資料庫載入成功！數量: {vectorstore._collection.count()}")
    else:
        print(f"警告：找不到 {DB_PATH} 資料夾，請確認已上傳至 GitHub")
except Exception as e:
    print(f"Chroma 載入失敗: {e}")

# ... (中間的 get_rag_context, SYSTEM_CHARACTER, call_llm 都不用變) ...
# ... (為了版面整潔，這裡省略中間邏輯，請保留你原本寫好的部分) ...
# ... (只展示最後啟動的部分) ...

# 設定 LINE API
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.route("/webhook", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # ... (保留你原本的邏輯) ...
    # 這裡只示範簡單結構
    user_id = event.source.user_id
    user_msg = event.message.text
    
    # 假設 call_llm 已經定義在上面
    # ai_reply = call_llm(user_id, user_msg) 
    
    # 這裡記得把你的 call_llm 邏輯放回來
    pass 

if __name__ == '__main__':
    # 修改：Render 會提供 PORT 環境變數，沒有的話預設 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)