
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR MultiAgents

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/web_demo.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
