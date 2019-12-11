# TF2.0 QA challenge submission

Repository for submission code for TF2.0 QA Challenge on [kaggle](https://www.kaggle.com/c/tensorflow2-question-answering/leaderboard)
Baseline model, last trained checkpoints and dataset converted to TF records available on [Google Drive](https://drive.google.com/open?id=1SNzHWpZuc_twnmjfBfyGardRO628AS2w)

## Prerequisites
Python setuptools and python package manager (pip) install packages into system directory by default.  The training code tested only via [virtual environment](https://docs.python.org/3/tutorial/venv.html).

In order to use virtual environment you should install it first:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv -p `which python3` <directory_for_environment>
```

Before starting to work inside virtual environment, it should be activated:

```bash
source <directory_for_environment>/bin/activate
```

Virtual environment can be deactivated using command

```bash
deactivate
```
Install dependencies with python package meneger:
```bash
pip install -r requirements.txt
```

## Baseline
BERT Baseline for NQ selected as baseline model. [[Original Repository](https://github.com/google-research/language/tree/master/language/question_answering/bert_joint)] [[article](https://arxiv.org/abs/1901.08634)]

## Data preprocessing
Training code can work with both original annotation format and preprocessed. Data preparation is time consuming process, I recommend to save it in TF records format for future experiments.

## Training
1. download baseline model to **bert-joint-baseline** directory from [Google Drive](https://drive.google.com/open?id=1SNzHWpZuc_twnmjfBfyGardRO628AS2w)
2. Download preprocessed data to **data** directory [Google Drive](https://drive.google.com/open?id=1SNzHWpZuc_twnmjfBfyGardRO628AS2w)

Following command run default baseline model training. For information about additional arguments please use `--help` option.
```bash
python main.py --bert_config_file bert-joint-baseline/bert_config.json \
--vocab_file bert-joint-baseline/vocab-nq.txt \
--train_precomputed_file data/nq-train.tfrecords \
--init_checkpoint bert-joint-baseline/bert_joint.ckpt
--do_train \
--output_dir bert_model_output \
--do_lower_case \
--train_num_precomputed 494670

```
