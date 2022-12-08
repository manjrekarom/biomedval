from typing import Any, List

import torch
import torchmetrics
import pytorch_lightning as pl
from torch.nn import functional as F


class BERTLightning(pl.LightningModule):
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
        self.model = 
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