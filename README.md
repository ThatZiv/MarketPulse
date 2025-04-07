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
- [Shadcn](https://ui.shadcn.com/docs/installation/vite) User-interface (UI) Library
- [TailwindCSS](https://tailwindcss.com/)

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

# Requirements

- [Node.js](https://nodejs.org/en) >= v20.14.0 (or any LTS)
- [Node Package Manager](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) (NPM) >= v10.7.0
- [Python 3.10](https://www.python.org/downloads/)

> [!NOTE]
> Alternatively, you may skip any requirements above and opt for the Docker image run - install [Docker](https://docs.docker.com/engine/install/) and [Docker compose](https://docs.docker.com/compose/install/) - this is not _needed_ but is a nice to have for development in containers.

- [Supabase](https://supabase.com/docs/guides/getting-started) instance (free tier works) - for authentication
  - You can use Supabase's free tier but understand its [limitations](https://supabase.com/pricing). Alternatively you can [self host Supabase](https://supabase.com/docs/guides/self-hosting) entirely.
- [Reddit API Key](https://developers.reddit.com/docs/api) - _for social media sentiment_
- [Logo Dev API Key](https://docs.logo.dev/introduction) - _for stock ticker logo images_
- [DeepSeek GGUF](https://huggingface.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF) Large Language Model (LLM) - these can be found on [hugging face](https://huggingface.co/search/full-text?q=deepseek+distil&type=model). Currently, our implementation supports the DeepSeek Architecture so any [distillations](https://huggingface.co/search/full-text?q=deepseek+distil&type=model) will work.
