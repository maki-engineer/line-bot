import os
from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

load_dotenv()

line_bot_api = LineBotApi(os.environ["LINE_API_KEY"])
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])
app = Flask(__name__)

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
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=event.message.text)
    )
    print("返信が完了しました。")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)
