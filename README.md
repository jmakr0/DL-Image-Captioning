# DL-Image-Captioning
Project for "Competitive Problem Solving with Deep Learning" at the Hasso-Plattner Institute

## Setup
For the current setup please refer to the imgcap-README

## Docker

Build the image locally:

``docker build -f docker/Dockerfile . --tag image-captioning``

Run the container:

``docker run --rm -v "Your_INPUT_Directory_Path:/usr/src/app/data/input" -v "Your_OUTPUT_Directory_Path:/usr/src/app/data/output" image-captioning``

Upload the container to dockerhub:

``bash docker/push_to_dockerhub.sh``
