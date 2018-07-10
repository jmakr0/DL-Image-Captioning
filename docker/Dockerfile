FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update \
    && apt-get install -y wget python3 python3-pip \
    && apt-get clean \
    && pip3 install --upgrade pip

COPY . /usr/src/app
WORKDIR /usr/src/app

RUN wget https://owncloud.hpi.de/index.php/s/jevzhc3KjyPRheN/download -O glove.6B.50d_normalized.txt
RUN mkdir -p data/embedding && mv glove.6B.50d_normalized.txt data/embedding/

RUN pip3 install -r requirements.txt

CMD [ "python3", "./src/coca_train.py"]