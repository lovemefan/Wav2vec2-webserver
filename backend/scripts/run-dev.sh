#!/usr/bin/env bash
set -o errexit

apt-get update -y && apt-get install libsndfile1 -y

app="../"

echo "Installing dependencies"
python3 -m pip install --upgrade pip
pip3 install -r "${app}/requirements.txt" -i "https://pypi.tuna.tsinghua.edu.cn/simple"
pip3 install -e "${app}../fairseq_lib"
echo "Starting Sanic development server"
cd $app
python3 "routes/app.py"
