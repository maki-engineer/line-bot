import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import LLMPredictor, PromptHelper, ServiceContext, QuestionAnswerPrompt
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

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

# 荒れた学級を元に戻すには？
text = input()

# ユーザーの入力文に関連する部分を抽出し、プロンプトに追加した上でユーザーの入力文をChatGPTに渡す
response = query_engine.query(text)
print(response)
