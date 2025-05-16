#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda create -n webarena python=3.10 --yes
conda activate webarena

cd ../webarena/
pip install -r requirements.txt
pip install --upgrade transformers
pip install --upgrade openai
pip install numpy==1.26.4
playwright install
pip install -e .
cd ../code
pip install -r requirements.txt
pip install -r requirements_rag.txt
# bash fix_mount.sh
