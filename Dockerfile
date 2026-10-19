FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NLTK_DATA=/app/nltk_data
# xgboost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
# bake the NLTK sentence data into the image so cold starts never need the network
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')"
COPY prohori ./prohori
COPY api ./api
COPY frontend ./frontend
COPY models ./models
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
