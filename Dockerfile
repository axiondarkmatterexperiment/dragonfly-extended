# If you're updating the base image, please keep it on an explicit version, don't use "latest" or similar
# ... except that v1.14.2 isn't out and we need to pick up the new dripline-python version
#from project8/dragonfly:latest
from project8/dragonfly:v1.16.3-arm

RUN pip install PyModbusTCP
RUN pip install scipy

# going to try just installing this, if we need to carefully only include it in arm installs we'll figure that out later
#RUN pip install rpi.gpio Adafruit_ADS1x15

COPY ./source /usr/local/src
