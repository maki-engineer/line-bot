import os
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import LLMPredictor, PromptHelper, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
line_bot_api = LineBotApi(os.environ["LINE_API_KEY"])
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])
app = Flask(__name__)

# dataというフォルダ内のテキストファイル（原稿データ）をベクトルデータベース化
documents = SimpleDirectoryReader("data").load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# ユーザーの入力文を処理するエンジンの作成
query_engine = index.as_query_engine()

# 使用するLLMやパラメータをカスタマイズする
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# プロンプトのテンプレートをカスタマイズする
QA_PROMPT_TMPL = (
    "関連する情報は以下の通りです。"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "{query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

header = {
    "Content_type": "application/json",
    "Authorization": "Bearer" + os.environ["LINE_API_KEY"]
}


@app.route("/")
def hello_world():
    return "Hello world"


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    response = query_engine.query(event.message.text)
    response = str(response)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')
