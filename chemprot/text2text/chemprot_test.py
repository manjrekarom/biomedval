import csv
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer


parser = argparse.ArgumentParser(description='Generates validation summaries for evaluation   \
and writes them to a csv file.')
parser.add_argument('ckpt', type=str, help='Checkpoint to use for evaluation')
parser.add_argument('--name-prefix', type=str, default='base', help='Prefix of the file being saved')
parser.add_argument('--results-dir', type=str, default='./results', help='Folder to store the results')
parser.add_argument('--data', type=str, default='./data/test.text_cleaned.tsv', help='Path to test data')

args = parser.parse_args()
print('Validation args:\n', args)


# load config, tokenizer and model
# checkpoint = '/home/omanjrekar/checkpoints/nlp/t5-sume'
checkpoint = args.ckpt
t5_config = T5Config.from_pretrained(checkpoint)
print('Config:', t5_config)

t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
t5_model = T5ForConditionalGeneration.from_pretrained(checkpoint)

val_sume_frost = load_dataset('text', data_files=args.data, split='train')


greedy_results = f'./{args.results_dir}/dev_pred_{args.name_prefix}_greedy.csv'
greedy_results_clean = f'./{args.results_dir}/dev_pred_{args.name_prefix}_greedy.txt'
beam_results = f'./{args.results_dir}/dev_pred_{args.name_prefix}_beam.csv'
beam_results_clean = f'./{args.results_dir}/dev_pred_{args.name_prefix}_greedy.txt'


# greedy
with open(greedy_results, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['ref_summary', 'pred_summary'])
    for example in val_sume_frost['text']:
        context, ref_summary = example.strip().split('<exp>')
        encoded_inputs = t5_tokenizer('summarize: ' + context.strip(), return_tensors='pt')
        outputs = t5_model.generate(
                        **encoded_inputs,
                        min_length=64,
                        max_length=128,
                        eos_token_id=t5_tokenizer.eos_token_id,
                        early_stopping=True,
                        repetition_penalty=2.5,
                        temperature=0.7)
        outputs = t5_tokenizer.decode(outputs[0])
        # print(outputs)
        writer.writerow([ref_summary, outputs])


# beam
with open(beam_results, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['ref_summary', 'pred_summary'])
    for example in val_sume_frost['text']:
        context, ref_summary = example.strip().split('<exp>')
        encoded_inputs = t5_tokenizer('summarize: ' + context.strip(), return_tensors='pt')
        outputs = t5_model.generate(
                        **encoded_inputs, 
                        min_length=64, 
                        num_beams=5,
                        max_length=128,
                        early_stopping=True,
                        eos_token_id=t5_tokenizer.eos_token_id,
                        no_repeat_ngram_size=3)
        outputs = t5_tokenizer.decode(outputs[0])
        # print(outputs)
        writer.writerow([ref_summary, outputs])
