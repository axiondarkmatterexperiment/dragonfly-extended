## NOTE: these defaults are not used by travis!
## you *must* update the .travis.yml file to change the dripline-python version
ARG img_user=project8
ARG img_repo=dragonfly
ARG img_tag=v1.19.4

from ${img_user}/${img_repo}:${img_tag}

RUN pip3 install PyModbusTCP
# RUN pip3 install pyserial
#RUN pip3 install scipy
RUN apt-get update && apt-get install -y python3-scipy
RUN if (uname -a | grep arm); then pip3 install RPi.GPIO Adafruit_ADS1x15 ; fi

# going to try just installing this, if we need to carefully only include it in arm installs we'll figure that out later
#RUN pip install rpi.gpio Adafruit_ADS1x15

COPY ./source /usr/local/src
