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
```

**NOTE:** There's a bug which prevents the correct best model from being loaded at the end for evaluation. I am guessing the way to solve this is by using a new trainer with the saved t5_model. I haven't tried it though. For now please rely on running the eval only command shown below.

**Evaluate a t5 model**
```
```

There are other options available which you can find in the `minimal_train.py` script.
