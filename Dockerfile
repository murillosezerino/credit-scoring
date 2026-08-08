FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo (inclui models/ se um modelo treinado for commitado)
COPY . .

ENV PORT=8000
EXPOSE 8000

# Serve a API. /docs, /health e / funcionam sem modelo;
# /predict exige models/stacking_model.pkl (gerado por `python main.py`).
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
