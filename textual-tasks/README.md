# Installation

This repository has been tested with Python 3.10.14. We recommend using Conda for environment management. You can set up the Conda environment using the following commands:

```
conda create --name dccot python=3.10.14 -y
conda activate dccot
pip install -r requirements.txt
```


# Run Experiments

Please use the train.py that is assosciated with the type of method you would like to run.
### Args usages
* `--task`: which dataset to use, can be: `SQA`, `CSQA`, `ARC`, `MATH`, `GSM8K`, `TabMWP`, `ANLI`, `Date`.
* `--n`: the proportion of training data to use, can be `1-100`. Default: `100`.
* `--model`: the student model to train, can be: `mistral-7b`, `gemma-7b`, `llama-8b`, `llama-r1`, `qwen`, `qwen-smallest`, `qwen-small`, `qwen-medium`

### Extract data
All data files can be download the dataset: https://huggingface.co/datasets/rana-shahroz/DC-COT

We use the dataset Date as an example below.


### Run

**Make sure that the directory structure has the training files named correctly**

**Step 1:** Train the student model using the following script:
```
CUDA_VISIBLE_DEVICES=0 python train.py --task SQA --n 100 --model mistral-7b 
```

**Step 2:** Evaluate the student model with the following script:
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --task SQA --n 100 --model mistral-7b 
```