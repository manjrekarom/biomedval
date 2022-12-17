from datasets import load_dataset
from transformers import T5Tokenizer


dataset = load_dataset('text', data_files={'train': 'data/train.txt_cleaned.tsv', 'val': 'data/dev.txt_cleaned.tsv'})
# print(dataset['train'][0])
trainset = dataset['train']
valset = dataset['val']

# pandas datasets
def lookup_labels(trainset):
    trainset = trainset.with_format('pandas')
    labels = trainset['text'].map(lambda example: example.split('\t')[1])
    print(labels.value_counts())

lookup_labels(trainset)

checkpoint = 'razent/SciFive-base-Pubmed_PMC'
t5_tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# t5_tokenizer.add_special_tokens()

# preprocess dataset
def preprocess_function(examples, prefix="chemprot_re: "):
    # print(examples['context'])
    inputs, labels = zip(*[example.split('\t') for example in examples['text']])
    inputs = [prefix + doc.strip() for doc in inputs]
    dataset = t5_tokenizer(inputs, max_length=1024, truncation=True)

    with t5_tokenizer.as_target_tokenizer():
        labels = t5_tokenizer(labels, max_length=128, truncation=True)

    dataset["labels"] = labels["input_ids"]
    return dataset

# train_sume['train'] = train_sume['train'].map(preprocess_function, batched=True)
trainset = trainset.map(preprocess_function, remove_columns=['text'], batched=True)
valset = valset.map(preprocess_function, remove_columns=['text'], batched=True)

print(trainset[0])
print(trainset[100])
