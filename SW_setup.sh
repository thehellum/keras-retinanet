#!/bin/bash

# Colors
RD='\033[0;31m' # Red
BL='\033[0;34m' # Blue
NC='\033[0m'    # No Color

IN_DIR=$(pwd)



# Check if sudo
# if [ "$EUID" -ne 0 ]
#   then printf "${RD}[!] Please run as root (sudo).${NC}\n"
#   exit
# fi

clear

# -- return to home directory --
cd

# --- update ---

sudo apt-get update
sudo apt-get upgrade

# --- if this script has been obtained without git ---

sudo apt-get install git-core

# --- setup pip and virtual enviroment ---

sudo apt install python3-dev python3-pip
sudo pip3 install -U virtualenv  # system-wide install

virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate  # sh, bash, ksh, or zsh
pip install --upgrade pip
pip3 install --upgrade pip3

# --- install tensorflow ---

pip install --upgrade tensorflow
pip3 install --upgrade tensorflow

# --- install div ---

pip3 install numpy --user
pip3 install opencv-python --user
pip3 install matplotlib --user
pip3 install keras --user
pip3 install h5py --user
pip3 install pillow --user
pip3 install scipy --user
pip3 install filetype --user
pip3 install pandas --user

# --- install associated dependencies ---

cd IN_DIR

pip install . --user
pip3 install . --user