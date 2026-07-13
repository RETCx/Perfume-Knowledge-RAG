
FROM python:3.11-slim


WORKDIR /app


RUN pip install --no-cache-dir --upgrade pip


COPY requirements.txt .


RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


RUN pip install --no-cache-dir -r requirements.txt


COPY . .

# Pre-download HuggingFace embedding model during image build so it doesn't download on every Cloud Run startup
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"
EXPOSE 8080


CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]