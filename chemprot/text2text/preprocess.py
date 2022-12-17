import glob
import shutil
import random


train_pos_files = glob.glob('data/aclImdb/train/pos/*.txt')
train_neg_files = glob.glob('data/aclImdb/train/neg/*.txt')
print(len(train_pos_files), len(train_neg_files))

random.shuffle(train_pos_files)
random.shuffle(train_neg_files)

val_pos_files = train_pos_files[:1000]
val_neg_files = train_neg_files[:1000]

# for f in val_pos_files:
#     shutil.move(f, 'data/aclImdb/val/pos')

# for f in val_neg_files:
#     shutil.move(f, 'data/aclImdb/val/neg')
