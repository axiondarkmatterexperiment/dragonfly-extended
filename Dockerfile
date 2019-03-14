# If you're updating the base image, please keep it on an explicit version, don't use "latest" or similar
from project8/dragonfly:v1.14.1

RUN pip install PyModbusTCP

# going to try just installing this, if we need to carefully only include it in arm installs we'll figure that out later
RUN pip install rpi.gpio

COPY ./source /usr/local/src
