#!/usr/bin/env bash

# Apply docker

docker login -u dlgroup8
docker build --tag image-captioning -f docker/Dockerfile .
docker tag image-captioning dlgroup8/image-captioning:latest
docker push dlgroup8/image-captioning
docker logout