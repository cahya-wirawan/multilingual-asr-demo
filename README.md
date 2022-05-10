# Multilingual Indonesian ASR Demo


* [Getting started](#getting-started)

## Getting started
There are two ways to run the demo, without Docker and using Docker.

Clone this repo:
```
git clone https://github.com/cahya-wirawan/multilingual-asr-demo.git
```

```
cd multilingual-asr-demo
```

### Without Docker
First, install the requirements:

```
pip install -r requirements.txt
```
then run the gradio:
```
python app.py
```

The website will be run in `localhost:7860`.

### Docker
Build the docker image:
```
docker build -t multilingual-asr-demo .
```
then run the image:
```
docker run -p 7860:7860 multilingual-asr-demo
```

The web application will run in `localhost:7860`.

