#!/bin/bash

## Step 1: Install the original ArXiv dataset from Google Drive
fileid="1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC"
filename="arxiv-dataset.zip"
curl -c /tmp/gd_cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb /tmp/gd_cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' /tmp/gd_cookie`&id=${fileid}" -o "${filename}"

## Step 2: Install unzip if applicable and unzip the file
sudo apt-get -y install unzip
unzip "${filename}"

# Step 3: Place the data in the correct directory
mv arxiv-dataset raw_data

# Step 4: Delete the zip file and the MACOS specific files that were included
rm "${filename}"
rm -r ./__MACOSX
