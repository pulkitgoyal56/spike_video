# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

RUN apt -y update
RUN apt -y upgrade
RUN apt install -y ffmpeg

RUN pip3 install jupyterlab

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY src ./
COPY notebooks/application.ipynb .

RUN mkdir data .temp

ENTRYPOINT ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]