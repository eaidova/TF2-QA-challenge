# TF2.0 QA challenge submission

Repository for submission code for TF2.0 QA Challenge on [kaggle](https://www.kaggle.com/c/tensorflow2-question-answering/leaderboard)
Baseline model, last trained checkpoints and dataset converted to TF records available on [Google Drive](https://drive.google.com/open?id=1SNzHWpZuc_twnmjfBfyGardRO628AS2w)

## Prerequisites
python >= 3.5
tensorflow >= 2.0

## Baseline
BERT Baseline for NQ selected as baseline model. [[Original Repository](https://github.com/google-research/language/tree/master/language/question_answering/bert_joint)] [[article](https://arxiv.org/abs/1901.08634)]

## Data preprocessing
Training code can work with both original annotation format and preprocessed. Data preparation is time consuming process, I recommend to save it in TF records format for future experiments.

## Training
Following command run default baseline model training. For information about additional arguments please use `--help` option.
```bash
python main.py --bert_config_file bert-joint-baseline/bert_config.json \
--vocab_file bert-joint-baseline/vocab-nq.txt \
--train_precomputed_file nq-train.tfrecords \
--init_checkpoint bert-joint-baseline/bert_joint.ckpt
--do_train \
--output_dir bert_model_output \
--do_lower_case \
--train_num_precomputed 494670
```
