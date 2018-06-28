FROM nvidia/cuda

RUN apt-get update \
    && apt-get install -y python-pip python-dev build-essential \
    && apt-get clean \
    && pip install --upgrade pip

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN pip install -r requirements.txt
CMD [ "python", "./run.py"]