import os
import csv
import argparse

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

import utils
from seqeval.metrics import f1_score, accuracy_score, classification_report, recall_score, precision_score


parser = argparse.ArgumentParser(description='Generates validation summaries for evaluation   \
and writes them to a csv file.')
parser.add_argument('--use-ckpt', type=str, help='Checkpoint to use for evaluation')
parser.add_argument('--name', type=str, default='default', help='Prefix of the file being saved')
parser.add_argument('--results-dir', type=str, default='./results', help='Folder to store the results')
parser.add_argument('--data', type=str, default='data/test.tsv_cleaned.tsv', help='Path to validation data')
parser.add_argument('--eval-only', action='store_true', help='Don\'t generate')

args = parser.parse_args()
print('Validation args:\n', args)

# create results directory
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# load config, tokenizer and model
# checkpoint = '/home/omanjrekar/checkpoints/nlp/t5-sume'
checkpoint = args.use_ckpt
t5_config = T5Config.from_pretrained(checkpoint)
print('Config:', t5_config)

t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
t5_model = T5ForConditionalGeneration.from_pretrained(checkpoint)


# preprocess dataset
# def preprocess_function(examples, prefix="bc5cdr_chem_ner: "):
#     # print(examples['context'])
#     inputs, labels = zip(*[example.split('\t') for example in examples['text']])
#     inputs = [prefix + doc.strip() for doc in inputs]
#     # TODO: 
#     dataset = t5_tokenizer(inputs, max_length=256, truncation=True)

#     with t5_tokenizer.as_target_tokenizer():
#         # TODO:
#         labels = t5_tokenizer(labels, max_length=256, truncation=True)

#     dataset["labels"] = labels["input_ids"]
#     return dataset


testset = load_dataset('text', data_files=args.data, split='train')

greedy_results = f'./{args.results_dir}/{args.name}_greedy.csv'
beam_results = f'./{args.results_dir}/{args.name}_beam.csv'

# greedy
if not args.eval_only:
    print('Generating greedy...')
    with open(greedy_results, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'output'])
        for example in testset['text']:
            text, label = example.strip().split('\t')
            # TODO: maybe try max_length for tokenizer and truncation true
            encoded_inputs = t5_tokenizer('bc5cdr_chem_ner: ' + text.strip(), return_tensors='pt')
            outputs = t5_model.generate(
                            **encoded_inputs,
                            min_length=64,
                            max_length=256,
                            eos_token_id=t5_tokenizer.eos_token_id,
                            early_stopping=True,
                            repetition_penalty=2.5,
                            temperature=0.3)
            outputs = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(outputs)
            writer.writerow([label, outputs])
    print(f'Greedy results are saved at {greedy_results}!')
    print()

    # beam
    print('Generating beam...')
    with open(beam_results, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'output'])
        for example in testset['text']:
            text, label = example.strip().split('\t')
            # TODO: maybe try max_length for tokenizer and truncation true
            encoded_inputs = t5_tokenizer('bc5cdr_chem_ner: ' + text.strip(), return_tensors='pt')
            outputs = t5_model.generate(
                            **encoded_inputs, 
                            min_length=64, 
                            num_beams=3,
                            max_length=256,
                            early_stopping=True,
                            eos_token_id=t5_tokenizer.eos_token_id,
                            no_repeat_ngram_size=3)
            outputs = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(outputs)
            writer.writerow([label, outputs])
    print(f'Beam results are saved at {beam_results}!')

# all about metrics
def convert2bio(decoded_preds, decoded_labels, debug=True):
    if debug:
        print('Eval preds length:', len(preds))
        print('First 4 preds values:', decoded_preds[:4])
        print('First 4 target values:', decoded_labels[:4])

    # Some simple post-processing
    decoded_preds, decoded_labels = utils.postprocess_text(decoded_preds, decoded_labels)
    dropped_idxs = []
    final_preds = []
    final_labels = []
    for i in range(len(decoded_labels)):
        try:
            final_pred = utils.convert_bio_labels(decoded_preds[i], select_only=['chem'])
            final_label = utils.convert_bio_labels(decoded_labels[i], select_only=['chem'])
            len_pred = len(final_pred)
            len_label = len(final_label)
            if len_pred > len_label:
                final_pred = final_pred[:len_label]
            else:
                final_pred = final_pred + ['PAD'] * (len_label - len_pred)
            final_preds.append(final_pred)
            final_labels.append(final_label)
        except Exception as e:
            print(e)
            dropped_idxs.append(i)
            # pass
    return final_preds, final_labels, dropped_idxs


# evaluate against 
print('Running evaluate...')
df = pd.read_csv(greedy_results)
preds, labels = df['output'].tolist(), df['label'].tolist()
bio_preds, bio_labels, dropped_idxs = convert2bio(preds, labels, debug=True)
print('Greey Gen. Report:', classification_report(bio_labels, bio_preds))

df = pd.read_csv(beam_results)
preds, labels = df['output'].tolist(), df['label'].tolist()
bio_preds, bio_labels, dropped_idxs = convert2bio(preds, labels, debug=True)
print('Beam Gen. Report:', classification_report(bio_labels, bio_preds))
print('\nDone!')
