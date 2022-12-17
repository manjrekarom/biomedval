# README #

Evaluate pre-trained transformer on several biomedical tasks

1. Relation Extraction
   - ChemProt
2. Named Entity Recognition
   - BC5CDR-Chem
   - BC5CDR-Disease
3. Semantic Similarity
   - BIOSSES


### 1. BlueBERT
The detailed instructions on running the train/eval are present in the [repository](https://github.com/ncbi-nlp/bluebert.git)


Example 1 for NER:
```
BlueBERT_DIR="checkpoints/NCBI_BERT_pubmed_uncased"
DATASET_DIR="../bert_data/BC5CDR/chem/"
OUTPUT_DIR="checkpoints/NCBI_BERT_pubmed_uncased-BC5CDR-chem/"

python -m bluebert.run_bluebert_ner \
  --do_prepare=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --task_name="bc5cdr" \
  --vocab_file=$BlueBERT_DIR/vocab.txt \
  --bert_config_file=$BlueBERT_DIR/bert_config.json \
  --init_checkpoint=$BlueBERT_DIR/bert_model.ckpt \
  --num_train_epochs=30.0 \
  --do_lower_case=true \
  --data_dir=$DATASET_DIR \
  --output_dir=$OUTPUT_DIR
```

### 2. BioBERT
See the instructions in this [repository](https://github.com/dmis-lab/biobert-pytorch)

### 3. BioBERT, ClinicalBERT, SciBERT, BlueBERT
See the instructions in this [repository](https://github.com/sy-wada/blue_benchmark_with_transformers.git)

### 4. T5
Checkout subfolders bc5cdr for NER and chemprot for Rel-Ext.
