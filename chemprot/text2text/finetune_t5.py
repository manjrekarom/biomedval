import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
from datasets import load_dataset

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


set_seed(42)
logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))


args_dict = dict(
    data_dir="", # path for data files
    output_dir="", # path to save the checkpoints
    model_name_or_path='razent/SciFive-base-Pubmed_PMC',
    tokenizer_name_or_path='razent/SciFive-base-Pubmed_PMC',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)


# datasets
trainset = 'data/train.txt'
valset = 'data/dev.txt'

# load datasets
train_chemprot = load_dataset('text', data_files=trainset, split='train')
val_chemprot = load_dataset('text', data_files=valset, split='train')
print(train_chemprot.features)
print(val_chemprot.features)

# load config, tokenizer and model
checkpoint = 'razent/SciFive-base-Pubmed_PMC'
# t5_config = T5Config.from_pretrained(checkpoint)
# print('Config:', t5_config)

t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
special_tokens = {'additional_special_tokens': ['chemprot_re:']}
t5_tokenizer.add_special_tokens(special_tokens)

t5_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
t5_model.resize_token_embeddings(len(t5_tokenizer))
# tokenized_inputs = t5_tokenizer(inputs, return_tensors='pt')
data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=t5_model, 
label_pad_token_id=t5_tokenizer.pad_token_id)


# preprocess dataset
prefix = "summarize: "
def preprocess_function(examples):
    # print(examples['context'])
    contexts, summaries = zip(*[ex.strip().split('<exp>') for ex in examples['text']])
    summaries = ['<exp> ' + summ.strip() for summ in summaries]
    inputs = [prefix + doc.strip() for doc in contexts]
    model_inputs = t5_tokenizer(inputs, max_length=1024, truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(summaries, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# train_sume['train'] = train_sume['train'].map(preprocess_function, batched=True)
train_chemprot = train_chemprot.map(preprocess_function, remove_columns=['text'], batched=True)
val_chemprot = val_chemprot.map(preprocess_function, remove_columns=['text'], batched=True)
