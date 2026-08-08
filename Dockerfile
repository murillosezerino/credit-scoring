FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo (inclui models/serving_model.pkl e scaler.pkl, commitados para o demo)
COPY . .

ENV PORT=8000
EXPOSE 8000

# Serve a API. models/serving_model.pkl e scaler.pkl ja vem no repo,
# entao /predict funciona out-of-the-box.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
