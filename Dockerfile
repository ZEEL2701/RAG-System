FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY RagApplication/requirements.txt /app/RagApplication/requirements.txt

# CPU-only PyTorch keeps the image smaller (embeddings still work the same)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/RagApplication/requirements.txt

COPY RagApplication /app/RagApplication

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=false

EXPOSE 7860

CMD ["python", "-m", "RagApplication"]
