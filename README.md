# MarketPulse

<center>

<img src="https://i.imgur.com/fYOR4Uz.png" height=200 weight=200>

</center>

# Introduction

MarketPulse is a web-based application built to empower both beginner and experienced investors by simplifying market analysis and decision-making. Recognizing that navigating the complexities of the stock market, with its overwhelming data and intricate trends, can be particularly challenging for newcomers, MarketPulse aims to bridge this gap. The platform enables users to select specific stocks and gain access to price predictions alongside actionable insights like buy or sell recommendations.

Powered by machine learning algorithms, MarketPulse continuously monitors a wide range of factors—social media activity, news reports, and major events—to identify trends that could influence stock prices. This data is presented through intuitive charts visualizing market sentiment and projected price movements, providing clear guidance for engaging with the stock market. To further enhance understanding, MarketPulse also incorporates an AI Large Language Model (LLM) capable of synthesizing all relevant information into a concise summary for each investment opportunity.

# Overview

<details>
<summary>Client</summary>

React-based (fully client) web application following a standard model-view-controller (MVC) pattern. Technologies used:

- React v19
- [Vite](https://vite.dev/) v6.0.5
- [React Query](https://tanstack.com/query/latest) for caching and handling requests
- [React Router](https://reactrouter.com/guides/home) for internal routing
- [Shadcn](https://ui.shadcn.com/docs/installation/vite) User-interface (UI) Library
- [TailwindCSS](https://tailwindcss.com/) for additional styling

</details>
<details>
<summary>Server</summary>

Server is a standard API-Gateway interface. Authentication is still managed by Supabase (via [JWT](https://supabase.com/docs/guides/auth/jwts)) will middleware.

- Python 3.10
- [Flask](https://flask.palletsprojects.com/en/stable/) v3.1.0
- [SQLAlchemy](https://www.sqlalchemy.org/) v2.0.37 - Object-relational mapping (ORM) to interface with Database
- [PyTorch](https://pytorch.org/) v2.6.0 - for timeseries forecasting
- [LangChain](https://python.langchain.com/docs/introduction/) v0.3.19 - for LLM integration and prompting
- PostgreSQL (Supabase-provided) - Database of choice
- _Additional third-party services mentioned in [requirements](#requirements)_

</details>

## Architecture

The architecture of MarketPulse is designed to be modular and scalable. The client and server are separated into two distinct directories, each with its own set of dependencies and configurations. The client is a React-based web application that communicates with the server via RESTful APIs. The server is a Flask-based API gateway that handles authentication, data processing, and machine learning model forecasts.

### Data flow

<details>
<summary>View</summary>

![dfd-diagram](https://i.imgur.com/Qu5t4WY.png)

</details>

### Use Case

<details>
<summary>View</summary>

![use-case-diagram](https://i.imgur.com/mSqHViN.png)

</details>

### Security

<details>
<summary>View</summary>

![sec-diagram](https://i.imgur.com/tvhs9Rn.png)

</details>

### Software

<details>
<summary>View</summary>

![software-diagram](https://i.imgur.com/yDmfciQ.png)

</details>

### Communication

<details>
<summary>View</summary>

![comm-diagram](https://i.imgur.com/1w57u8w.png)

</details>

### Sequence (User Authentication)

<details>
<summary>View</summary>
![sequence-diagram](https://i.imgur.com/dnpxYyQ.png)
</details>

## Structure

<details>
<summary>Open file hierarchy</summary>

```

├── server/
│   ├── data/
│   ├── database/ -- DB ORM methods
│   ├── public/
│   │   └── cache/ -- static files served
│   ├── routes/
│   │   ├── auth.py -- protected routes for logos, forecast, etc.
│   │   └── llm.py -- protected route for LLM output
│   ├── models/
│   │   ├── forecast/ -- where all forecast models implementations are/
│   │   │   ├── azad.py -- SARIMA/ARIMA forecast implementation
│   │   │   ├── attention_lstm.py -- AttentionLSTM forecast implementation
│   │   │   ├── cnn_lstm.py -- CNN LSTM forecast implementation
│   │   │   ├── transformer.py -- Transformer forecast implementation
│   │   │   ├── xgboost.py -- XGBoost forecase implementation
│   │   │   ├── model.py -- Base class for forecast models
│   │   │   └── models.py -- Base class to manage ALL forecast models implementations
│   │   ├── zav2.py -- transformer base class
│   │   ├── xgboost_model.py -- XGBoost base class
│   │   ├── sentiment_hf.py -- sentiment analysis
│   │   ├── run_models.py -- runs all base model implementations
│   │   └── lstm_attension.py -- LSTM base class
│   ├── test_data/
│   ├── main.py -- entry point (with some additional routes)
│   ├── engine.py -- database pooler
│   ├── stockdataload.py -- load stock data from yahoo into db
│   ├── .env.example
│   └── .env -- you must copy this from .env.example
├── client/
│   ├── cypress/ -- e2e (system) tests
│   ├── public/
│   ├── src/
│   │   ├── __test__/ -- unit and integration tests
│   │   ├── components/ -- components directory/
│   │   │   ├── ui/ -- shadcn component library
│   │   │   ├── llm/
│   │   │   │   └── stock-llm.tsx -- handle llm output in StockEstimation
│   │   │   ├── recommendation-chart.tsx -- aside from the name, this calcs recommendation (StockEstimation)
│   │   │   ├── buy_sell.tsx -- used in recommendation-chart.tsx for buy/sell recommendation
│   │   │   ├── radial-chart.tsx -- used for impact and hype meter (StockEstimation)
│   │   │   ├── historical-chart.tsx -- historical prices chart (StockEstimation)
│   │   │   ├── historical-accuracy.tsx -- Forecast model calculator implementation
│   │   │   └── <tons of other components>
│   │   ├── hooks/
│   │   ├── database/
│   │   │   ├── supabase.ts -- Supabase wrapper
│   │   │   └── SupabaseProvider.tsx -- Global user auth provider (supabase client impl.)
│   │   ├── lib/
│   │   │   ├── api.ts -- API wrapper for backend
│   │   │   ├── ApiProvider.tsx -- API wrapper implementation
│   │   │   ├── Calculator.ts -- Purchase history and Forecast calculators
│   │   │   ├── constants.ts -- used in caching and reducer actions
│   │   │   ├── dataHandler.ts -- used for client caching (used in queries)
│   │   │   ├── GlobalProvider.tsx -- Global app state managed here (including reducer logic)
│   │   │   └── utils.ts -- reusable and misc functions
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx -- aside from the name, main view AFTER authenticated
│   │   │   ├── Landing.tsx -- home page (index) AFTER authenticated '/'
│   │   │   ├── Settings.tsx -- settings page '/settings'
│   │   │   ├── Documentation.tsx -- documentation page '/documentation'
│   │   │   ├── StockSelection.tsx -- stock selection page '/stocks'
│   │   │   ├── StockEstimation.tsx -- stock analysis page '/stock/:ticker'
│   │   │   ├── UserAuth.tsx -- default catch all when not authenticated '/auth'
│   │   │   └── <more pages and subpages>
│   │   ├── types/
│   │   ├── App.tsx --routing goes through here
│   │   └── main.tsx -- entrypoint
│   ├── .env.example
│   ├── .env.local -- you must copy this from .env.example
│   └── <other files>
└── supabase/
    ├── migrations/
    │   └── <list of database changes to migrate to your own database individually>
    └── schema.sql -- this is a schema dump that can be solely imported to your data (recommended)
```

</details>

## Environment Variables

<details>
<summary>Client</summary>

| Environment Variable    | More Info                                                                                                                            | Example                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| `VITE_API_URL`          | The URL location of the backend python webserver.                                                                                    | `http://localhost:5000`                   |
| `VITE_SUPABASE_URL`     | The Supabase-provided project URL                                                                                                    | `https://xyz.supabase.co`                 |
| `VITE_SUPABASE_KEY`     | The Supabase-provided project key. More info found [here](https://supabase.com/docs/guides/api/api-keys).                            | `eyJhbGciOiJIUzB1NiIaInB5cCI2IklXVCJ9...` |
| `VITE_GOOGLE_CLIENT_ID` | The Google Client ID provided by GCP. More info can be found [here](https://supabase.com/docs/guides/auth/social-login/auth-google). | `abc123xyz456`                            |

</details>

<details>
<summary>Server</summary>

| Environment Variable | More Info                                                                                                                                           | Example                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `SUPABASE_URL`       | The Supabase-provided project URL                                                                                                                   | `https://xyz.supabase.co`                 |
| `SUPABASE_KEY`       | The Supabase-provided project key                                                                                                                   | `eyJhbEciOiJIUzC1NiIaInB2cCI2LklXVCJ9...` |
| `SUPABASE_JWT`       | The Supabase-provided JWT secret key that can be found in the [Supabase API settings](https://app.supabase.com/project/_/settings/api)              | `abc123xyz456`                            |
| `reddit_secret_key`  | The Reddit API secret key. You can create one [here](https://business.reddithelp.com/s/article/Create-a-Reddit-Application)                         | `abc123xyz456`                            |
| `reddit_public_key`  | The Reddit API public key. You can create one [here](https://business.reddithelp.com/s/article/Create-a-Reddit-Application)                         | `abc123xyz456`                            |
| `LOGODEV_API_KEY`    | The Logo Dev API key for stock ticker images. You can create one [here](https://docs.logo.dev/introduction)                                         | `pk_d64204492aeb0b297461d9de2`            |
| `LLM_MODEL_PATH`     | The path to the LLM model. You can find the model [here](https://huggingface.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF)              | `DeepSeek-R1-Distill-Qwen-1.5B-GGUF`      |
| `user`               | The username for the PostgreSQL database                                                                                                            | `postgres`                                |
| `password`           | The password for the PostgreSQL database                                                                                                            | `abc123xyz456`                            |
| `host`               | The host for the PostgreSQL database                                                                                                                | `localhost`                               |
| `port`               | The port for the PostgreSQL database                                                                                                                | `5432`                                    |
| `dbname`             | The database name for the PostgreSQL database                                                                                                       | `postgres`                                |
| `LEGACY`             | The legacy flag for the PostgreSQL database. This only runs the webserver and everything that doesn't use tensorflow (no models nor scheduled jobs) | `true` (default is `false`)               |

</details>

# Dependencies

MarketPulse utilizes a wide array of third-party services to source it's data from.

# Requirements

- [Node.js](https://nodejs.org/en) >= v20.14.0 (or any LTS)
- [Node Package Manager](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) (NPM) >= v10.7.0
- [Python 3.10](https://www.python.org/downloads/)

> [!NOTE]
> Alternatively, you may skip any requirements above and opt for the Docker image run - install [Docker](https://docs.docker.com/engine/install/) and [Docker compose](https://docs.docker.com/compose/install/) - this is not _needed_ but is a nice to have for development in containers.

- [Supabase](https://supabase.com/docs/guides/getting-started) instance (free tier works) - for authentication
  - You can use Supabase's free tier but understand its [limitations](https://supabase.com/pricing). Alternatively you can [self host Supabase](https://supabase.com/docs/guides/self-hosting) entirely.
- [Reddit API Credentials](https://developers.reddit.com/docs/api) - _for social media sentiment_
- [Logo Dev API Key](https://docs.logo.dev/introduction) - _for stock ticker logo images_
- [DeepSeek GGUF](https://huggingface.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) Large Language Model (LLM) - these can be found on [hugging face](https://huggingface.co/search/full-text?q=deepseek+distil&type=model). Currently, our implementation supports the DeepSeek Architecture so any [distillations](https://huggingface.co/search/full-text?q=deepseek+distil&type=model) will work.
- [Google Client ID](https://supabase.com/docs/guides/auth/social-login/auth-google) - used for authentication by Google.
- [RoBERTa](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) transformer model for sentiment analysis - this is used to analyze the sentiment of the news articles and social media posts. _No action is needed to install this - the server should install it on-demand._

# Setup

1. Install/setup all requirements from the [requirements](#requirements) section.
2. Clone the repo

```sh
git clone https://github.com/ThatZiv/MarketPulse/
cd MarketPulse
```

## Client setup

3. Navigate to client directory

```sh
cd client
```

4. Install dependencies

```sh
npm install
```

5. Copy `.env.example` to `.env.local`. Please fill out all the corresponding values as mentioned in the [environment variables](#environment-variables) section.

6. Run the server

```sh
npm run dev # to run DEV server
```

> [!NOTE]
> Alternatively, you can run the server in a production environment:
>
> ```sh
> npm run build # build PROD server
> npx serve dist/ # run basic webserver serving the build folder
> # OR
> python -m http.server dist/ # run basic webserver with python
> ```

## Server setup

7. Navigate to server directory

```sh
# from the root directory
cd server
```

8. Create a virtual environment (venv)

```sh
python -m venv ./.venv

source .venv/bin/activate # for linux/mac
.venv\Scripts\activate # for windows

```

9. Install dependencies

```sh
pip install -r requirements.txt

# you may have to install some additional dependencies manually
sudo apt-get install libpq-dev postgresql-client # for linux

```

10. Copy `.env.example` to `.env`. Please fill out all the corresponding values as mentioned in the [environment variables](#environment-variables) section.

11. Run the server

```sh
python main.py # to run the server
```

## Supabase setup

Instructions will differ if you opt to self-host Supabase or use the free tier. For the free tier, you can read the [Supabase documentation](https://supabase.com/docs/guides/getting-started) to get situated.

12. Create a new project in Supabase and set up the database. You can use the default settings for the database.

13. Import the `supabase/schema.sql` into your Supabase database. This will create the necessary tables and functions for the application to work. This can be done multiple ways:
    - Using the [Supabase SQL editor](https://supabase.com/features/sql-editor)
    - Using the [Supabase CLI](https://supabase.com/docs/guides/cli)
    - Directly using the [PostgreSQL CLI](https://www.postgresql.org/docs/current/app-psql.html) (if you have it installed locally)
      - ```sh
        psql -h <host> -U <user> -d <dbname> < supabase/schema.sql # using the same variables from the [environment variables](#environment-variables) section.
        ```

## Docker setup

Alternatively, you can run the entire application using Docker. This will build the client and server images and run them in containers.

# Contributing

Contributions are welcomed and encouraged! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information on how to contribute to this project.

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

# Contact/Support

If you would like to contact us, please reach out to us via [email](mailto:zavaar.shah@wayne.edu).
