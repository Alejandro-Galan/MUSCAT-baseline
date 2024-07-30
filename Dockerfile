FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive


RUN apt update --fix-missing
RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 -y
RUN apt install vim -y
RUN apt install fluidsynth -y
RUN apt install git -y
RUN apt clean

RUN pip install --upgrade pip
RUN pip install pybind11

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN apt install libfuse2
