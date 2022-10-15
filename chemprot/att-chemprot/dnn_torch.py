"""
Attention based Recurrent Neural Network for biomedical relation extraction within a sentence.

The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

The implementation and IO is based on :
https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2017-07_Seminar/Session%203%20-%20Relation%20CNN

Code was tested with:
- Python 2.7
- TensorFlow 1.2.1
- Keras 2.0.5
"""

import os
import sys
import gzip

import torch
import numpy as np
from torch.utils.data import random_split
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, confusion_matrix

from att_lstm_torch import LSTMAttLightning
from annot_util.config import ChemProtConfig


if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

np.random.seed(42)  # for reproducibility

config = ChemProtConfig('config/main_config.ini')

# CNN hyperparameters
batch_size = 64
# nb_filter = 200
# filter_length = 3
nb_epoch = 20
position_dims = 50
dropout_rate = 0.5
lstm_units = 128
learning_rate = 0.001
weights = 1/0.3
class_weights = {0:1/0.7, 1:weights, 2:weights, 3:weights, 4:weights, 5:weights}

mode = 'ent_candidate'

# choose between 'cnn', 'gru' ,'att_lstm', 'att_gru'

model_name = 'att_lstm'

model_dir = config.get('main', 'model_dir')

# load prepared data in .pkl the same ways as preprocess.py
pkl_path = 'pkl/bioc_rel_%s.pkl.gz' % mode
root_dir = 'data/'
fns = ['training.txt', 'development.txt', 'test.txt']
files = [os.path.join(root_dir, fn) for fn in fns]
print("mode: " + mode)

gs_dev_txt = files[1]
gs_test_txt = files[2]

print("Loading dataset")
# f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
f = gzip.open(pkl_path, 'rb')

# data = pkl.load(f, encoding='latin1')
data = pkl.load(f)

f.close()

embeddings = data['wordEmbeddings']
y_train, sentence_train, position_train1, position_train2 = data['train_set']
y_dev, sentence_dev, position_dev1, position_dev2 = data['dev_set']
y_test, sentence_test, position_test1, position_test2 = data['test_set']

max_position = max(np.max(position_train1), np.max(position_train2)) + 1

n_out = max(y_train) + 1

max_sentence_len = sentence_train.shape[1]

print("sentenceTrain: ", sentence_train.shape)
print("positionTrain1: ", position_train1.shape)
print("yTrain: ", y_train.shape)

print("sentenceDev: ", sentence_dev.shape)
print("positionDev1: ", position_dev1.shape)
print("yDev: ", y_dev.shape)


# stack training with dev
# comment out  the following four lines if you would like to train models only on the training set
y_train = np.hstack((y_train, y_dev))
sentence_train = np.vstack((sentence_train, sentence_dev))
position_train1 = np.vstack((position_train1, position_dev1))
position_train2 = np.vstack((position_train2, position_dev2))


target_names = config.get_target_labels()

max_sentence_len = max(sentence_train.shape[1], sentence_dev.shape[1])

print("class weights:")
print(class_weights)


def predict_classes(prediction, pred_tag=''):
    # save probabilities for dev set
    if pred_tag != '':
        np.savetxt('output/pred_prob_%s_%s.txt' % (pred_tag, model_name), prediction, fmt="%.5f")
    return prediction.argmax(axis=-1)


def load_model(model_dir):
    model = LSTMAttLightning.load_from_checkpoint(model_dir)
    model.eval()
    return model


def init_att_lstm_model():
    model = LSTMAttLightning(hidden_dim=lstm_units, output_dim=n_out, ent1_max_dist=max_position, 
    ent2_max_dist=max_position, ent1_dist_dim=position_dims, ent2_dist_dim=position_dims, 
    weights=embeddings, class_weights=class_weights)
    return model


def do_training():
    """
    Main function of training DNN models
    :return:
    """

    init_func = {
        # 'cnn': init_cnn_model,
        # 'gru': init_att_gru_model,
        # 'att_gru': init_att_gru_model,
        'att_lstm': init_att_lstm_model,
        # 'att_rnn': init_att_rnn_model,
    }


    # tensor dataset
    train_tensors = []
    for np_arr in [sentence_train, position_train1, position_train2, y_train]:
        train_tensors.append(torch.tensor(np_arr))
    
    train_dataset = TensorDataset(*train_tensors)
    
    # split train into train and val
    train_size = int(0.85 * len(y_train))
    val_size = len(y_train) - train_size
    
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size,])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    # model
    model = init_func[model_name]()

    # trainer
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint_cb = ModelCheckpoint(monitor="val_f1_score", mode="max", save_top_k=2)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=30, callbacks=[early_stopping_cb, checkpoint_cb])
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Training done. Model saved at: {checkpoint_cb.best_model_path}")
    return checkpoint_cb.best_model_path


def do_test(model_path, stage='test'):
    """
    Run official submission
    :param stage: 'test' or 'dev'.
    :return:
    """
    print("##" * 40)

    print('Stage: %s. starting evaluating using %s set: ' % (stage, stage))

    model = load_model(model_path)

    if stage == 'dev':
        y_gs = y_dev
        pred = predict_classes(model(sentence_dev, position_dev1, position_dev2))
        gs_txt = gs_dev_txt
    elif stage == 'test':
        global sentence_test, position_test1, position_test2, y_test

        test_tensors = []
        for np_arr in [sentence_test, position_test1, position_test2, y_test]:
            test_tensors.append(torch.tensor(np_arr))
        test_dataset = TensorDataset(*test_tensors)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

        out = []
        for batch in test_loader:
            sentence, pos1, pos2, _ = batch
            sentence = sentence.to('cuda:0')
            pos1 = pos1.to('cuda:0')
            pos2 = pos2.to('cuda:0')
            out.append(model(sentence, pos1, pos2).detach().cpu().numpy())
        
        out = np.concatenate(out, axis=0)
        y_gs = y_test

        pred = predict_classes(out, pred_tag='bilstm-att-last-layer')
        gs_txt = gs_test_txt

    else:
        raise ValueError("Unsupported stage. Requires either \"dev\" or \"test\".")

    output_tsv = config.get(stage, 'output_tsv')
    gs_tsv = config.get(stage, 'gs_tsv')

    # official eval has different working directory (./eval)

    write_results(os.path.join('eval', output_tsv), gs_txt, pred)
    official_eval(output_tsv, gs_tsv)

    print()
    print('Confusion Matrix: ')
    print(confusion_matrix(y_gs, pred))

    print()
    print('Classification Report:')
    print(classification_report(y_gs, pred, labels=range(1, 6),
                                target_names=target_names[1:],
                                digits=3))

    return pred


def write_results(output_tsv, gs_path, pred):
    """
    Write list of output in official format
    :param output_tsv:
    :param pred:
    :return:
    """
    ft = open(gs_path)
    lines = ft.readlines()

    assert len(lines) == len(pred),  'line inputs does not match: input vs. pred : %d / %d' % (len(lines), len(pred))

    with open(output_tsv, 'w') as fo:
        for pred_idx, line in zip(pred, lines):
            splits = line.strip().split('\t')
            if target_names[pred_idx] == "NA":
                continue

            fo.write("%s\t%s\tArg1:%s\tArg2:%s\n" %
                     (splits[-1], target_names[pred_idx],
                     splits[-3], splits[-2],
                     ))
    print("results written: " + output_tsv)
    ft.close()


def official_eval(output_tsv, gs_tsv):
    """
    Run official evaluation
    :param output_tsv:
    :param gs_tsv:
    :return:
    """
    print()
    print('Official Evaluation Results:')
    os.chdir('eval')
    os.system("./eval.sh %s %s" % (output_tsv, gs_tsv))
    os.chdir('..')
    print()


if __name__ == '__main__':
    best_model_path = do_training()
    # model_path = './lightning_logs/version_2/checkpoints/epoch=5-step=2346.ckpt'
    # model_path = './lightning_logs/version_7/checkpoints/epoch=6-step=2737.ckpt'
    # model_path = './lightning_logs/version_7/checkpoints/epoch=8-step=3519.ckpt'
    # model_path = './lightning_logs/version_8/checkpoints/epoch=4-step=1370.ckpt'
    # model_path = './lightning_logs/version_8/checkpoints/epoch=5-step=1644.ckpt'
    do_test(best_model_path, stage='test')
    # do_test(model_path, stage='test')
