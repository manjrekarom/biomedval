import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score

import spacy
import scispacy
import evaluate
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Config, \
    T5ForConditionalGeneration, T5Tokenizer

# argparse
parser = argparse.ArgumentParser(description='SuMe training using HF Trainer API')

parser.add_argument('--trainset', type=str, default='data/train.txt_cleaned.tsv', help='ChemProt training set')
parser.add_argument('--valset', type=str, default='data/dev.txt_cleaned.tsv', help='ChemProt dev set')
parser.add_argument('--testset', type=str, default='data/test.txt_cleaned.tsv', help='ChemProt testing set')
parser.add_argument('--use-ckpt', type=str, default=None, help='Use ckpt for eval. Works when --eval-only is set')
parser.add_argument('--eval-only', action='store_true', help='Only predict/evaluate')
parser.add_argument('--use-both', action='store_true', help='Use both train and val for training')
parser.add_argument('--name', type=str, default='default', help='Experiment name')
parser.add_argument('--batch-size', type=int, default=4, help='Train and eval batch size')
parser.add_argument('--num-epochs', type=int, default=30, help='No. of epochs')
parser.add_argument('--max-steps', type=int, default=-1, help='Max steps (overrides --num-epochs)')

args = parser.parse_args()
print('Training Args: ', args)

# load spacy model for sentence tokenization
# core_sci_sm_nlp = spacy.load('en_core_sci_sm')

# load datasets
# trainset = load_dataset('text', data_files='data/train.txt_cleaned.tsv', split='train[:512]')
if args.use_both:
    trainset = load_dataset('text', data_files={"train": [args.trainset, args.valset]}, split='train')
    valset = load_dataset('text', data_files=args.testset, split='train')
else:
    trainset = load_dataset('text', data_files=args.trainset, split='train')
    # valset = load_dataset('text', data_files='data/dev.txt_cleaned.tsv', split='train[:128]')
    valset = load_dataset('text', data_files=args.valset, split='train')
testset = load_dataset('text', data_files=args.testset, split='train')

print(trainset.features)
print(valset.features)
print(testset.features)

# pandas datasets
def lookup_labels(dataset, splitted=False):
    dataset = dataset.with_format('pandas')
    if not splitted:
        labels: pd.Series = dataset['text'].map(lambda example: example.split('\t')[1])
    else:
        labels 
    return labels

labels = lookup_labels(trainset)
print('Labels in ChemProt: ', labels.value_counts())
label_encoder = LabelEncoder().fit(labels)
print('Label encoder classes:', label_encoder.classes_)
test_labels = sorted(label_encoder.transform(lookup_labels(testset).value_counts().index))
print('Test labels:', test_labels)

# load config, tokenizer and model
# checkpoint = '/home/omanjrekar/checkpoints/nlp/t5-sume'
default_ckpt = 'razent/SciFive-base-Pubmed_PMC'
checkpoint = default_ckpt if not args.use_ckpt else args.use_ckpt
print(f'Using checkpoint {checkpoint}!!!')
t5_config = T5Config.from_pretrained(checkpoint)

t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# TODO: Maybe add labels as special tokens
# special_tokens = {'additional_special_tokens': ['<exp>', '<re>', '<er>', '<el>', '<le>']}
# t5_tokenizer.add_special_tokens(special_tokens)

t5_model = T5ForConditionalGeneration.from_pretrained(checkpoint)
# t5_model.resize_token_embeddings(len(t5_tokenizer))
# tokenized_inputs = t5_tokenizer(inputs, return_tensors='pt')
data_collator = DataCollatorForSeq2Seq(tokenizer=t5_tokenizer, model=t5_model, 
label_pad_token_id=t5_tokenizer.pad_token_id)

# preprocess dataset
def preprocess_function(examples, prefix="chemprot_re: "):
    # print(examples['context'])
    inputs, labels = zip(*[example.split('\t') for example in examples['text']])
    inputs = [prefix + doc.strip() for doc in inputs]
    # TODO: 
    dataset = t5_tokenizer(inputs, max_length=256, truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        # TODO:
        labels = t5_tokenizer(labels, max_length=15, truncation=True)

    dataset["labels"] = labels["input_ids"]
    return dataset

# train_sume['train'] = train_sume['train'].map(preprocess_function, batched=True)
trainset = trainset.map(preprocess_function, remove_columns=['text'], batched=True)
valset = valset.map(preprocess_function, remove_columns=['text'], batched=True)
testset = testset.map(preprocess_function, remove_columns=['text'], batched=True)

# all about metrics
# preprocess text for metrics
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

# compute metrics)
# accuracy = evaluate.load("accuracy")
# f1 = evaluate.load("f1")
# precision = evaluate.load("precision")
# recall = evaluate.load("recall")
dropped = 0

def le_transform(preds, targets):
    le_preds = []
    le_targets = []
    dropped_batch = 0
    for i in range(len(preds)):
        try:
            le_preds.append(label_encoder.transform([preds[i]])[0])
            le_targets.append(label_encoder.transform([targets[i]])[0])
        except ValueError:
            dropped_batch += 1
    print('Dropped batches:', dropped_batch)
    return le_preds, le_targets, dropped_batch


def convert2le(preds, labels, tokenizer, debug=False):
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    if debug:
        print('Eval preds length:', len(preds))
        print('First 25 preds values:', decoded_preds[:25])
        print('First 25 target values:', decoded_labels[:25])

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    return le_transform(decoded_preds, decoded_labels)


# bleurt_metric = load_metric("bleurt")
def compute_metrics(eval_preds):
    global dropped
    preds, labels = eval_preds
    batch_size = len(labels)
    le_preds, le_labels, dropped_batch = convert2le(preds, labels, t5_tokenizer, debug=True)
    print('First 25 le_preds values:', le_preds[:25])
    print('First 25 le_labels values:', le_labels[:25])
    labels = range(len(label_encoder.classes_))

    dropped += dropped_batch / batch_size
    p, r, f, _ = precision_recall_fscore_support(y_pred=le_preds, y_true=le_labels, average='macro')
    result = {'precision': p, 'recall': r, 'f1': f}
    result['drop_rate'] = dropped
    # prediction_lens = [np.count_nonzero(pred != t5_tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    # classification report
    # print('Report: ', classification_report(y_pred=le_preds, y_true=le_labels, digits=4))
    return result

# train loop
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./checkpoints/{args.name}",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    # TODO:
    # 1e-3
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    greater_is_better=True,
    # load best model at the end
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.num_epochs,
    max_steps=args.max_steps,
    fp16=False,
    predict_with_generate=True,
    gradient_accumulation_steps=4,
)

trainer = Seq2SeqTrainer(
    model=t5_model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=valset,
    tokenizer=t5_tokenizer,
    # TODO: 
    # optimizer use Adafactor
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if not args.eval_only:
    trainer.train()
    print(f'Saving the best model to "./checkpoints/{args.name}/best/"')
    trainer.save_model(f"./checkpoints/{args.name}/best/")


# perform evaluation
preds, labels, metrics = trainer.predict(testset)
print('\n\n** Metrics **\n')
print(metrics)
le_preds, le_labels, dropped = convert2le(preds, labels, t5_tokenizer, debug=True)
print('Report:', classification_report(y_true=le_labels, y_pred=le_preds, digits=4,
    labels=test_labels, target_names=label_encoder.classes_))
p, r, f, support = precision_recall_fscore_support(y_pred=le_preds, y_true=le_labels, average='micro')
print(f'Micro precision: {p}, recall: {r}, f1: {f}, support: {support}')
