#!/bin/bash
fileid="1b3rmCSIoh6VhD4HKWjI4HOW-cSwcwbeC"
filename="arxiv-dataset.zip"
curl -c /tmp/gd_cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb /tmp/gd_cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' /tmp/gd_cookie`&id=${fileid}" -o "${filename}"
