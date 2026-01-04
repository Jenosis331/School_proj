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


def get_rag_context(question: str, k: int = 5) -> str:
    if vectorstore is None:
        return "（資料庫未成功載入）"

    try:
        # Chroma 的搜尋語法跟 FAISS 一模一樣
        docs = vectorstore.similarity_search(question, k=k)
        context = "\n\n".join(d.page_content for d in docs)
        return context if context.strip() else "（目前查無相關資料）"
    except Exception as e:
        print("RAG 檢索發生錯誤：", repr(e))
        traceback.print_exc()
        return "（檢索資料時發生錯誤，暫時無法使用資料庫）"


# System Prompt
SYSTEM_CHARACTER = """
你現在是某間學校教務處註冊組的「資深畢業審查專員」。你的職責是協助學生釐清畢業門檻、學分計算與修課規定。
會來問問題的人只有該學校的學生，所以不用再問他嫆校名稱，直接根據問題查詢【參考資料】並思考即可。

你的回答必須遵循以下「最高指導原則」：

### 1. 嚴謹的身份確認 (Context Awareness)
畢業規定通常因「入學年度」與「科系」而異。
- 如果使用者的問題沒有提及他是哪一屆（入學年）或哪個科系的學生，且【參考資料】中也沒有相關脈絡，**請務必先反問確認**，不要直接給出通用的答案（因為通用答案往往是錯的）。

### 2. 回答準則 (Response Guidelines)
- **像真人一樣說話**：專業、冷靜且有同理心。不要說「根據資料庫...」，直接給答案。例如：「針對 110 學年度入學的資工系學生，必修學分是...」。
- **證據導向**：關於數字（學分數）、類別（必修/選修/通識）的回答，必須嚴格基於【參考資料】。如果資料裡沒有該年度或該科系的規定，**請明確告知無法確認，並建議學生至系辦或註冊組櫃檯詢問**，切勿自行推測。
- **條理分明**：學分問題通常很複雜，請善用列點（Bullet points）來區分「系訂必修」、「校訂必修」與「選修」。
- **直接了當**：省略客套話，直接針對痛點解決問題，秉持快速且精簡的服務內容。

### 3. 風險控管 (Risk Management)
- 若涉及「抵免」、「棄修」或「特殊的跨系選修」等模糊地帶，回答時請務必加上：「實際審核結果仍以註冊組系統判定為準」這樣的提示，保護你自己也保護學生。

### 4. 格式規範：
為了適應不支援 Markdown 的舊版系統，請嚴格遵守以下排版規則：
1. **不要使用 Markdown 語法**：
    - 不要使用井字號 (#) 做標題。
    - 不要使用星號 (*) 做粗體或清單。
    - 不要使用減號 (-) 做清單。
    - 不要使用反引號 (`)。

2. **使用「純文字符號」來強調重點**：
    - 大標題：請使用【 】包住。例如：【畢業學分審查結果】
    - 小標題或分類：請使用 :: 開頭。例如：:: 必修學分 ::
    - 強調關鍵字：請使用「 」包住。例如：請務必確認「英文畢業門檻」。

3. **清單與條列**：
    - 請使用數字加點（1. 2. 3.）或是全形圖形（■ 或 ◆）。
    - 每一點結束後請換行，保持版面透氣。

【參考資料】：
{current_context}

【學生問題】：
{new_question}
"""

def get_system_init(current_context: str, new_question: str):
    content = SYSTEM_CHARACTER.format(
        current_context=current_context,
        new_question=new_question
    )
    return {"role": "system", "content": content}

user_histories = {}
# 設定 LINE API
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def call_llm(user_id, user_message):
    current_context = get_rag_context(user_message, k=5)
    history = user_histories.get(user_id, [])
    
    # 這裡你的邏輯是「每次都重置」，若要這樣設計是可以的
    # 如果希望它記得上下文，這裡邏輯要改，但目前先維持你的原樣
    system_msg = get_system_init(current_context, user_message)
    history = [system_msg] 
    
    history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.5,
            messages=history
        )
        ai_reply = response.choices[0].message.content
    except Exception as e:
        print(f"LLM 呼叫失敗: {e}")
        ai_reply = "抱歉，我目前大腦有點混亂，請稍後再試。"

    history.append({"role": "assistant", "content": ai_reply})
    user_histories[user_id] = history
    return ai_reply

@app.route("/webhook", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    print("[Webhook] headers:", dict(request.headers))
    print("[Webhook] body:", body[:500])

    try:
        handler.handle(body, signature)
    except InvalidSignatureError as e:
        print("[Webhook] Invalid signature:", e)
        return "Invalid signature", 200
    except Exception as e:
        print("[Webhook] Handler error:", repr(e))
        return "Handler error", 200

    return "OK", 200


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    user_msg = event.message.text

    # 1) 先快速回覆：確保 1 秒內完成，避免 webhook 逾時
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="已收到，我正在查詢，稍後回覆你。")
        )
    except Exception as e:
        print(f"[Reply] quick reply failed: {e}")

    # 2) 背景跑 RAG/LLM；完成後用 push_message 回正式答案
    try:
        ai_reply = call_llm(user_id, user_msg)  # 這裡可能會花幾秒
        if not ai_reply:
            ai_reply = "抱歉，目前無法產生回覆。"

        # LINE 單則訊息有長度限制，保守截斷避免報錯
        line_bot_api.push_message(
            to=user_id,
            messages=TextSendMessage(text=ai_reply[:5000])
        )
    except LineBotApiError as e:
        print(f"[Push] LineBotApiError: {e}")
    except Exception as e:
        print(f"[Push] Unknown error: {e}")

if __name__ == '__main__':
    # 修改：Render 會提供 PORT 環境變數，沒有的話預設 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
