import os
from flask import Blueprint, Response, request
import flask_jwt_extended as jw
from langchain_community.llms import LlamaCpp
from flask_cors import cross_origin
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

llm_bp = Blueprint('llm', __name__, url_prefix='/llm')

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")

# DOWNLOAD FROM: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
# OR https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/tree/main


background = "\
You are an expert financial assistant for an website called MarketPulse. \
You are to only speak directly to the user and never address them in the third person. \
Avoid using jargon, technical terms, and markdown in your responses. \
Keep your responses short and concise and avoid rambling. Only answer in text, no need for lists or headings. \
You are to present to user the options to either, buy, sell, or hold based on current data \
MarketPulse is a stock trading application that gives users the ability to make \
financial decisions based on the latest news and trends. \
The app provides real-time updates on stock prices, market trends, and financial news to help users make informed decisions \
about their investments and trades. \
Your role is simply provide the user with a summarization about the stock they requested \
with the following information. \
"

@llm_bp.route('/stock', methods=['GET'])
@jw.jwt_required()
def llm__stock_route():
    if LLM_MODEL_PATH is None:
        print("LLM_MODEL_PATH is not set")
        return Response(status=500)

    if not os.path.exists(LLM_MODEL_PATH):
        print("LLM_MODEL_PATH file does not exist. Please download a gguf model from https://huggingface.co/models")
        return Response(status=500)
    current_user = jw.get_jwt_identity()
    if current_user is None:
        return Response(status=401)
    print(current_user)
    ticker = request.args.get('ticker')
    if not ticker:
        return "Ticker parameter is required", 400
    # TODO: use these params
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=1,
        temperature=0.75,
        max_tokens=4096,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        n_batch=206,
        n_ctx=2048,
        f16_kv=True,
        # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
    )

    def generate_response():
        # FIXME: make this prompt integration better
        for chunk in llm.stream(background + \
            " User currently has 250 shares of Apple stock. Current sentiment shows that it's forecasting to decrease in value. What should the user do?"):
            yield chunk
    # def generate_response():
        # is_thinking = True # needed in deepseek models :(
        # for chunk in llm.stream(background + \
        #     " User currently has 250 shares of Apple stock. Current sentiment shows that it's forecasting to decrease in value. What should the user do?"):
        #     print(chunk)
        #     # FIXME: break when request is cancelled
        #     if not is_thinking:
        #         # we dont'w want to show our thinking
        #         yield chunk
        #     if is_thinking and "</think>" in chunk:
        #         is_thinking = False

    return Response(generate_response(), content_type='text/event-stream')
