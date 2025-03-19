# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=global-statement

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, pool
import time
from flask import current_app
ENGINE = None

run_args = {
    # "client_encoding": "utf8",
    "poolclass": pool.SingletonThreadPool,
}


def global_engine():
    global ENGINE
    if ENGINE is None:
        with current_app.config["MUTEX"]:
            if ENGINE is None:
                engine = get_engine()
                pass
                return engine
            else:
                pass
                return ENGINE
    else:
        return ENGINE

# allows us to reset engine
def get_engine():
    global ENGINE
    retrys = 5
    for _ in range(retrys):
        load_dotenv()
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        dbname = os.getenv("dbname")

        database=f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"

        # ENGINE = create_engine(database, **run_args)
        try:
            engine = create_engine(database,
            pool_recycle=3600)
            with engine.connect():
                print("Connection successful!")
                break
        except exc.OperationalError as e:
            time.sleep(5)
            print(e)
        except exc.TimeoutError as e:
            time.sleep(5)
            print(e)
    ENGINE = engine
    return engine

    
