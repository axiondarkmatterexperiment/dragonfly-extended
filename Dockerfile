## NOTE: these defaults are not used by travis!
## you *must* update the .travis.yml file to change the dripline-python version
ARG img_user=project8
ARG img_repo=dragonfly
ARG img_tag=v1.19.4

from ${img_user}/${img_repo}:${img_tag}

RUN pip3 install PyModbusTCP
RUN pip3 install pyserial
#RUN pip3 install scipy
RUN sed -i -e 's/deb.debian.org/archive.debian.org/g' \
           -e 's|security.debian.org|archive.debian.org/|g' \
           -e '/stretch-updates/d' /etc/apt/sources.list &&\
    apt-get update &&\
    apt-get install -y python3-scipy
RUN if (uname -a | grep arm); then install_for_sag.sh; fi
#RUN if (uname -a | grep arm); then pip3 install --upgrade setuptools ; fi
#RUN if (uname -a | grep arm); then pip3 install RPi.GPIO Adafruit_ADS1x15 ; fi
# adding local src directory to pythonpath to simplify imports
ENV PYTHONPATH="${PYTHONPATH}:usr/local/src"
# going to try just installing this, if we need to carefully only include it in arm installs we'll figure that out later
#RUN pip install rpi.gpio Adafruit_ADS1x15

COPY ./source /usr/local/src
