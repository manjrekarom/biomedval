from symbol import parameters
from typing import Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LSTMAtt(nn.Module):
    def __init__(self, hidden_dim, output_dim, ent1_max_dist, ent2_max_dist, ent1_dist_dim=50, 
                 ent2_dist_dim=50, n_layers=1, use_bidirectional=False, use_dropout=False,
                 weights=None, vocab_size=None, embedding_dim=None, device='cpu'):
        self.device = device
        super().__init__()
        if weights is not None:
            weights = torch.DoubleTensor(weights)
            vocab_size = weights.shape[0]
            embedding_dim = weights.shape[1]
            self.word_embedding = nn.Embedding.from_pretrained(weights, freeze=True)  #vocab_size, embedding_dim)
        else:
            self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        # TODO: handle padding for distance embeddings
        self.distance_emb1 = nn.Embedding(ent1_max_dist, ent1_dist_dim)
        self.distance_emb2 = nn.Embedding(ent2_max_dist, ent2_dist_dim)
        # TODO: handle lstm no. of layers
        self.rnn = nn.LSTM(embedding_dim + ent1_dist_dim + ent2_dist_dim, 
                           hidden_dim // 2 if use_bidirectional else hidden_dim,
                           bidirectional=use_bidirectional,
                           dropout=0.5 if use_dropout else 0.,
                           num_layers=n_layers, device=self.device)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5 if use_dropout else 0.)

    def attention_net(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        # originally this was here; but I am not sure how it works for more than 1 layer
        # merged_state = torch.cat([s for s in final_state], 1)
        # TODO: try max, mean pool and just sum over merged state as query vector
        # below line implements max pooling
        merged_state, _ = torch.max(final_state, axis=0)
        # print('merged state', merged_state.shape)
        weights = torch.bmm(lstm_output, merged_state.unsqueeze(2))
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, word, dist1, dist2):
        word_embs = self.word_embedding(word)
        # print('word_embs', word_embs.shape)
        dist1_embs = self.distance_emb1(dist1)
        # print('dist1_embs', dist1_embs.shape)
        dist2_embs = self.distance_emb2(dist2)
        # print('dist2_embs', dist2_embs.shape)
        x = torch.cat((word_embs, dist1_embs, dist2_embs), axis=-1)  # batch_size, len, 
        x = torch.as_tensor(x, dtype=torch.double, device=self.device).permute(1, 0, 2)  # seq, batch_size, emb_dim
        # print('x dtype', x.dtype)
        output, (hidden, cell) = self.rnn(x)
        # print('output', output.shape)
        # print('hidden', hidden.shape)
        # attn_output = self.attention_net(output, hidden)
        attn_output = self.attention(output, hidden)
        # print('attn_output', attn_output.shape)

        return self.fc(attn_output.squeeze(0))


class LSTMAttLightning(pl.LightningModule):
    # CAUTION: mind the defaults; they might override model defaults
    def __init__(self, hidden_dim, output_dim, ent1_max_dist, ent2_max_dist, ent1_dist_dim=50, 
                 ent2_dist_dim=50, n_layers=1, use_bidirectional=False, use_dropout=False,
                 weights=None, vocab_size=None, embedding_dim=None, lr=1e-3, beta_1=0.9, 
                 beta_2=0.999, epsilon=1e-08, decay=0.0, class_weights: List[float] = None) -> None:
        super().__init__()
        # TODO: BUG; override device
        override_device = 'cuda:0'

        # needed for saving hyper parameters in the checkpoint
        self.save_hyperparameters()

        # model
        self.model = LSTMAtt(hidden_dim, output_dim, ent1_max_dist, ent2_max_dist, ent1_dist_dim=ent1_dist_dim, 
        ent2_dist_dim=ent2_dist_dim, n_layers=n_layers, use_bidirectional=use_bidirectional, use_dropout=use_dropout, 
        weights=weights, vocab_size=vocab_size, embedding_dim=embedding_dim, device=override_device) 
        self.model.double()
        self.model.to(device=override_device)

        # adam optimizer params
        self.lr = lr
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # class weights
        if class_weights is not None:
            self.class_weights = torch.as_tensor(list(class_weights.values()), dtype=torch.double, device=override_device)
        else:
            self.class_weights = torch.ones(output_dim, dtype=torch.double, device=override_device)
    
        # metrics
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.val_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.val_recall = torchmetrics.Recall()
        self.train_f1_score = torchmetrics.F1Score()
        self.val_f1_score = torchmetrics.F1Score()

    def forward(self, word, dist1, dist2) -> Any:
        return self.model(word, dist1, dist2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2), 
        eps=self.epsilon, weight_decay=self.decay)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        word, dist1, dist2, label = train_batch
        pred = self.model(word, dist1, dist2)
        loss = F.cross_entropy(pred, label.long(), weight=self.class_weights)
        self.log_dict({
            'train_loss': loss.item(),
            'train_acc': self.train_acc(pred, label),
            'train_precision': self.train_precision(pred, label),
            'train_recall': self.train_recall(pred, label),
            'train_f1_score': self.train_f1_score(pred, label),
        })
        return loss

    def validation_step(self, val_batch, batch_idx):
        word, dist1, dist2, label = val_batch
        pred = self.model(word, dist1, dist2)
        loss = F.cross_entropy(pred, label.long(), weight=self.class_weights)
        _, pred_classes = torch.max(pred, axis=1)
        # self.log('val_loss', loss)
        # acc = (pred_classes == label).float().sum() / pred.shape[0]
        # self.log('val_accuracy', acc)
        # self.val_confmat(pred, label)
        self.log_dict({
            'val_loss': loss.item(),
            'val_acc': self.val_acc(pred, label),
            'val_precision': self.val_precision(pred, label),
            'val_recall': self.val_recall(pred, label),
            'val_f1_score': self.val_f1_score(pred, label),
        })


if __name__ == "__main__":
    vocab_size, embedding_dim, hidden_dim, output_dim = 10, 4, 4, 6
    ent1_max_dist, ent2_max_dist = 3, 3
    lstm = LSTMAtt(hidden_dim, output_dim, ent1_max_dist, ent2_max_dist, n_layers=2,
    vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    batch_size = 4
    max_len = 7
    words = torch.randint(low=0, high=vocab_size, size=(max_len, batch_size))
    dist1 = torch.randint(low=0, high=ent1_max_dist, size=(max_len, batch_size))
    dist2 = torch.randint(low=0, high=ent2_max_dist, size=(max_len, batch_size))
    out = lstm(words, dist1, dist2)
    assert out.shape == (batch_size, output_dim)
