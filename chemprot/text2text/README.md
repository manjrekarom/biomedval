# Text2Text ChemProt
Trains and evaluates a t5 model trained with ChemProt. As it is with t5 or Seq2Seq models, all the tasks are considered conditional generation tasks. For ChemProt specifically, the output sequence will be the relation type in text format.

## Install libraries

Install pytorch
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`

Install all other libraries
`pip install -r requirements.txt`

## Train and eval

**Train a t5 model**
```
python minimal_train.py --trainset data/scifive-ds/mod_finetune_chemprot_train.txt_cleaned.tsv --valset data/scifive-ds/mod_finetune_chemprot_dev.txt_cleaned.tsv --testset data/scifive-ds/mod_finetune_chemprot_test.txt_cleaned.tsv  --batch-size 32 --use-ckpt ../../../t5-sume/ --num-epochs 30 --name t5-sume-chemprot-bs32-ep30 --use-both
```

**NOTE:** There's a bug which prevents the correct best model from being loaded at the end for evaluation. I am guessing the way to solve this is by using a new trainer with the saved t5_model. I haven't tried it though. For now please rely on running the eval only command shown below.

**Evaluate a t5 model**
```
python minimal_train.py --trainset data/scifive-ds/mod_finetune_chemprot_train.txt_cleaned.tsv --valset data/scifive-ds/mod_finetune_chemprot_dev.txt_cleaned.tsv --testset data/scifive-ds/mod_finetune_chemprot_test.txt_cleaned.tsv  --batch-size 32 --use-ckpt ../../../t5-sume/ --num-epochs 30 --name t5-sume-chemprot-bs32-ep30 --use-both --eval-only
```

There are other options available which you can find in the `minimal_train.py` script.
