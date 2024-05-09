FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y procps && pip install -U pip && rm /etc/localtime && ln -s /usr/share/zoneinfo/America/Mexico_City/

COPY ./requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY ./app.py /app.py

EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
