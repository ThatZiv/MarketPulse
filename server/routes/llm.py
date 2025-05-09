# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches

import os
from dotenv import load_dotenv
import flask_jwt_extended as jw
from flask import Blueprint, Response, request
# pylint: disable=no-name-in-module
from langchain_community.llms import LlamaCpp
# pylint: enable=no-name-in-module
# pylint: disable=line-too-long
# pylint: disable=not-callable
# from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from sqlalchemy import select
from langchain.prompts import PromptTemplate
from database.tables import Stocks, Stock_Info, User_Stock_Purchases
from routes.access import get_forcasts, create_session
from cache import cache

llm_bp = Blueprint('llm', __name__, url_prefix='/llm')
load_dotenv()
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH")

# DOWNLOAD FROM: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF
# OR https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/tree/main


SYSTEM_PROMPT = "\
You are an expert financial assistant for website called MarketPulse. Make sure to greet them. \
Avoid using jargon, technical terms, and markdown in your responses. You must only speak in english. \
Keep your responses short, concise, and avoid rambling. Only answer in clear text, no need for lists or headings. \
MarketPulse is a stock trading application that gives users the ability to make \
financial decisions based on the latest news and trends. \
The app provides real-time updates on stock prices, market trends, and financial news to help users make informed decisions \
about their investments and trades. \
Your role is simply provide the user with a summarization about the stock they requested \
with the following provided and recommend them to either, buy, sell, \
or hold based on the current context."

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
TEMPLATE = SYSTEM_PROMPT + "\n{query}"

prompt = PromptTemplate(template=TEMPLATE, \
    input_variables=['query'], \
    partial_variables={"system_prompt": SYSTEM_PROMPT})
@llm_bp.route('/stock', methods=['GET'])
@jw.jwt_required()
def llm__stock_route():
    if LLM_MODEL_PATH is None:
        print("LLM_MODEL_PATH is not set")
        return Response(status=500)

    if not os.path.exists(LLM_MODEL_PATH):
        print("LLM_MODEL_PATH file does not exist. \
            Please download a gguf model from https://huggingface.co/models")
        return Response(status=500)

    ticker = request.args.get('ticker')
    current_user = jw.get_jwt_identity()

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
    # TODO: get from database and refine the prompt iteself

    output = cache.get(f"LLM_{ticker}{current_user}")
    if output is None:
        session = create_session()
        s_id = select(Stocks).where(Stocks.stock_ticker == ticker)
        output_id = session.connection().execute(s_id).first()
        if output_id:
            stock_data = select(Stock_Info).where(Stock_Info.stock_id == output_id.stock_id).order_by(Stock_Info.time_stamp.desc()).limit(1)
            stock = session.connection().execute(stock_data).first()
            user_info = select(User_Stock_Purchases).where(User_Stock_Purchases.stock_id == stock.stock_id).where(User_Stock_Purchases.user_id == current_user)
            user_output = session.connection().execute(user_info).all()

            if not stock:
                return Response(status=500)
            output = {"output": stock, "user_info":user_output}
            cache.set(f"LLM_{ticker}{current_user}", output, timeout = 1800)
        else:
            return Response(status=500)

    stocks_owned = 0
    average = 0
    count = 0
    if output["user_info"]:
        for row in output["user_info"]:
            count+=1
            stocks_owned += row.amount_purchased
            average += row.amount_purchased*row.price_purchased
        if stocks_owned > 0:
            average = average/stocks_owned

    closing = output['output'].stock_close
    lookback = "1"
    response = cache.get("forecast_"+ticker+lookback)

    if response is None:
        response = get_forcasts(ticker=ticker, lookback=lookback)

    if response is None:
        return Response(status=500)

    if average !=0:
        query_template = f"Hello, I currently have shares of {ticker} stock. \
            I bought them for {average} dollars per share.\
            The current price is {closing} dollars.\
            I predict that tomorows price will be {response[0]['output'][-1]['forecast'][6]}\
            and next weeks will be {response[0]['output'][-1]['forecast'][6]}. What should I do?\n<think>\n"
    else :
        query_template = f"Hello, I currently have shares of {ticker} stock. \
            The current price is {closing} dollars.\
            I predict that tomorows price will be {response[0]['output'][-1]['forecast'][0]}\
            and next weeks will be {response[0]['output'][-1]['forecast'][6]}. What should I do?\n<think>\n"

    query = query_template
    def generate_response():
        """ stream llm response """
        #pylint: disable=use-yield-from
        for chunk in llm.stream(prompt.format(query=query)):
            # we handle thinking on the frontend
            # if "</think>" in chunk:
            #     yield "**Thinking complete.**\n"
            yield chunk

    return Response(generate_response(), content_type='text/event-stream')
