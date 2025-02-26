import os
from flask import Blueprint, Response, request
import flask_jwt_extended as jw
from langchain_community.llms import LlamaCpp
from flask_cors import cross_origin
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

llm_bp = Blueprint('llm', __name__, url_prefix='/llm')

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")

# DOWNLOAD FROM: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
# OR https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/tree/main


system_prompt = "\
You are an expert financial assistant for website called MarketPulse. Make sure to greet them. \
Avoid using jargon, technical terms, and markdown in your responses. You must only speak in english. \
Keep your responses short, concise, and avoid rambling. Only answer in clear text, no need for lists or headings. \
MarketPulse is a stock trading application that gives users the ability to make \
financial decisions based on the latest news and trends. \
The app provides real-time updates on stock prices, market trends, and financial news to help users make informed decisions \
about their investments and trades. \
Your role is simply provide the user with a summarization about the stock they requested \
with the following provided and recommend them to either, buy, sell, or hold based on the current context. \
<think>"

# for phi-3.5
# template = """<|system|>{system_prompt}<|end|>
# <|user|>
# {query}<|end|>
# <|assistant|>
# """
# for llama:
# template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\
# {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# for deepseek
template = system_prompt + "\n{query}"

prompt = PromptTemplate(template=template, \
    input_variables=['query'], \
    partial_variables={"system_prompt": system_prompt})
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
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=4,
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
        for chunk in llm.stream(prompt.format(query=f"Hello, I currently have 250 shares of {ticker} stock. What should I do?")):
            if "</think>" in chunk:
                yield "**Thinking complete.**\n"
            yield chunk
    # def generate_response():
    #     is_thinking = True # needed in deepseek models :(
    #     yield "Thinking about it...\n"
    #     for chunk in llm.stream(prompt.format(query=f"Hello, I currently have 250 shares of {ticker} stock. What should I do?")):
    #         # FIXME: break when request is cancelled
    #         if not is_thinking:
    #             # we dont'w want to show our thinking
    #             yield chunk
    #         if is_thinking and "</think>" in chunk:
    #             is_thinking = False

    return Response(generate_response(), content_type='text/event-stream')
