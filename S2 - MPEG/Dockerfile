FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]