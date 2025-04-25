# Stock Price Predictor

## What is it? 🤔

Predict stock prices by analyzing scraped news data with stock symbols references and their stock prices.

## Installation 💻

1. Python version 3.12 required.
2. Secondly, create a virtual environment for Python 3 with `python3.12 -m venv venv` and then activate with `source venv/bin/activate`.
3. Then, we need to install dependencies based on [pyproject.toml](pyproject.toml) file. Use `pip install uv` and then `uv pip install --upgrade eager -e .`.
4. Run `python -m spacy download en_core_web_trf`
⚠️ Be aware that if you use the existing requirements file to install dependencies, you may have compatibility issues due to different machines.

## Exclude from git large files 🚫

To avoid pushing files larger than 100 MB, use `find . -size +100M | cat >> .gitignore` and `find . -size +100M | cat >> .git/info/exclude`.

## Memory-Testing 🧠

It uses [memray](https://github.com/bloomberg/memray/blob/main/README.md) to profile memory. Very useful to optimize code and reduce memory usage thus <u>reducing google cloud costs</u>.

run in terminal :

```bash
python3 -m memray run -o data/static/html/memstats.bin src/handle_xls.py
```

Then, generate the flamegraph from the .bin file:
```
memray flamegraph data/static/html/memstats.bin
```

For more information on how to use see [Usage](https://github.com/bloomberg/memray/blob/main/README.md#usage).