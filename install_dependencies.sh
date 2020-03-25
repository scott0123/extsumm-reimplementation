#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    sudo apt -y install python3-dev python3-setuptools python3-pip htop
fi
sudo pip3 install --upgrade pip
sudo pip3 install numpy
sudo pip3 install torch
sudo pip3 install rouge
