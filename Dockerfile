FROM python:3.8-slim-buster
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt-get update && apt-get install -y sox libsndfile1
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 7860
CMD ["python", "app.py"]
