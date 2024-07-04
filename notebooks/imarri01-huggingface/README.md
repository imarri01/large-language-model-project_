---
license: apache-2.0
base_model: sshleifer/distilbart-cnn-12-6
tags:
- generated_from_trainer
model-index:
- name: imarri01-huggingface
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# imarri01-huggingface

This model is a fine-tuned version of [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) on the None dataset.
It achieves the following results on the evaluation set:
- eval_loss: 3.8916
- eval_rouge1: 0.2209
- eval_rouge2: 0.0472
- eval_rougeL: 0.1375
- eval_rougeLsum: 0.1894
- eval_runtime: 413.4048
- eval_samples_per_second: 0.121
- eval_steps_per_second: 0.017
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.42.3
- Pytorch 2.3.0
- Datasets 2.20.0
- Tokenizers 0.19.1
