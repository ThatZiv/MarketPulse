# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=global-statement

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, pool


ENGINE = None

run_args = {
    # "client_encoding": "utf8",
    "poolclass": pool.SingletonThreadPool,
}

def get_engine():
    global ENGINE
    if ENGINE is None:
        load_dotenv()
        user = os.getenv("user")
        password = os.getenv("password")
        host = os.getenv("host")
        port = os.getenv("port")
        dbname = os.getenv("dbname")

        database=f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?sslmode=require"

        # ENGINE = create_engine(database, **run_args)
        try:
            # ENGINE = create_engine(database, **run_args)
            # TODO: we dont have to close each connection whereever we do it
            ENGINE = create_engine(database)
            with ENGINE.connect():
                print("Connection successful!")
        except exc.OperationalError as e:
            print(e)
        except exc.TimeoutError as e:
            print(e)
        return ENGINE
    return ENGINE
