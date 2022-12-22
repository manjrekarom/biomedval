# Text2Text BC5CDR-Chem
Trains and evaluates a t5 model trained with BC5CDR-Chem dataset. As it is with t5 or Seq2Seq models, all the tasks are considered conditional generation tasks. For BC5CDR-Chem specifically, the output sequence will be the entire text with 1 or more pairs of "chem*" and "*chem" tags to identify.

## Install libraries

Install pytorch
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`

Install all other libraries
`pip install -r requirements.txt`

## Train and eval

**Train a t5 model**
```
CUDA_VISIBLE_DEVICES=0 python minimal_train.py --name default --batch-size 4
```

**Evaluate a t5 model**
```
CUDA_VISIBLE_DEVICES=0 python minimal_eval.py --name default --use-ckpt ./checkpoints/default/checkpoint-8550/ --results-dir ./results
```

There are other options available which you can find in the `minimal_train.py` script.
