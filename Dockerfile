## NOTE: these defaults are not used by travis!
## you *must* update the .travis.yml file to change the dripline-python version
ARG img_user=driplineorg
ARG img_repo=dripline-python
ARG img_tag=v3.10.1

from ${img_user}/${img_repo}:${img_tag}

RUN pip install PyModbusTCP
RUN pip install scipy

# going to try just installing this, if we need to carefully only include it in arm installs we'll figure that out later
#RUN pip install rpi.gpio Adafruit_ADS1x15

COPY ./source /usr/local/src
