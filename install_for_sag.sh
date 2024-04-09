#! /bin/bash

pip3 install --upgrade setuptools
pip3 install RPi.GPIO

# we have to install an old version of Adafruit Python ADS1x15 for compatibility with old dl2 dragonfly code
mkdir /usr/local/ada_src
cd /usr/local/ada_src
git clone https://github.com/adafruit/Adafruit_Python_ADS1x15.git
cd Adafruit_Python_ADS1x15
pip install .
