#! /bin/bash

pip3 install --upgrade setuptools
pip3 install RPi.GPIO

# we have to install an old version of Adafruit Python ADS1x15 for compatibility with old dl2 dragonfly code
apt-get install update
apt-get install -y git
mkdir /usr/local/ada_src
cd /usr/local/ada_src
git clone https://github.com/adafruit/Adafruit_Python_ADS1x15.git
pwd
ls
cd /usr/local/ada_src/Adafruit_Python_ADS1x15
pip3 install .
