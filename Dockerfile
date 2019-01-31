# If you're updating the base image, please keep it on an explicit version, don't use "latest" or similar
from project8/dragonfly:v1.14.1

RUN pip install PyModbusTCP

COPY ./source /usr/local/src
