from typing import Any, Dict, Optional, Union
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
from torch.utils.tensorboard.summary import hparams
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.metrics.functional import accuracy
from tqdm import tqdm
import sys
from .loss import MSE


class Model(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=1e-3, optim="Adam",zeroparams=None, 
                 regularization=None, regularizer=None):
        super().__init__()
        self.model = model
        self.Loss = criterion if criterion is not None else MSE(model.out_channels)
        self.optim = optim
        self.regularization = regularization
        self.regularizer = regularizer
        self.zeroparams = zeroparams
        if optim == "Adam":
            self.lr = lr
        else:
            self.lr = self.optim.defaults["lr"]
        try:
            nfilters = str(model.nfilters)
        except:
            nfilters = "N/A"

        self.hparams["params"] = sum([x.size().numel()
                                      for x in self.model.parameters()])
        #self.hparams["nfilters"] = nfilters
        self.hparams["loss"] = self.Loss
        self.hparams["optim"] = optim.__repr__().replace("\n","")
        self.hparams["model"] = self.model.__repr__().replace("\n", "")
        self.hparams["lr"] = self.lr
        if self.regularization is not None:
            self.hparams["regularizer"] = self.regularizer.__repr__()
            self.hparams["regularization"] = self.regularization
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, m, w = batch
        yhat = self(x)
        loss = self.Loss(yhat, y,w)
        self.log('train_loss', loss)
        
        if self.regularization is not None:
            if self.model.out_channels == 1:
                pred = torch.sigmoid(yhat).view(-1)
                loss_reg = self.regularization * self.regularizer(pred=pred, 
                                                           target=y.view(-1), 
                                                           x_biased=m.view(-1))
            else:
                pred = torch.softmax(yhat,dim=-1)
                loss_reg = self.regularization * self.regularizer(pred=pred[:,0], 
                                                           target=y.view(-1), 
                                                           x_biased=m.view(-1))
            loss += loss_reg
            self.log('train_loss_reg', loss_reg)
            self.log('train_loss_total', loss)
        if self.zeroparams is not None:
            self.log("zero_params", sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()]))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m, w = batch
        yhat = self(x)
        loss = self.Loss(yhat, y,w)
        if self.model.out_channels == 1:
            if self.model.readout_activation is None:
                preds = torch.sigmoid(yhat.view(-1,1))>.5
            else:
                preds = yhat.view(-1,1)>.5
            y = y.view(-1,1)
        else:
            preds = torch.argmax(yhat, dim=1)
        acc = accuracy(preds, y, class_reduction="weighted")
        # Calling self.log will surface up scalars for you in TensorBoard
        metrics = {'val_loss': loss, 'val_acc': acc}
        if self.regularization is not None:
            if self.model.out_channels == 1:
                pred = torch.sigmoid(yhat).view(-1)
                loss_reg = self.regularization * self.regularizer(pred=pred, 
                                                           target=y.view(-1), 
                                                           x_biased=m.view(-1))
            else:
                pred = torch.softmax(yhat,dim=-1)
                loss_reg = self.regularization * self.regularizer(pred=pred[:,0], 
                                                           target=y.view(-1), 
                                                           x_biased=m.view(-1))
            metrics['val_loss_reg'] =  loss_reg

        self.log_dict(metrics, prog_bar=True, logger=True,
                      on_epoch=True, on_step=False)
        try:
            self.logger.log_hyperparams(self.hparams, metrics=metrics)
        except:
            self.logger.log_hyperparams(self.hparams)
        return metrics

    def validation_epoch_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        val_loss_reg_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        metrics = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        
        if self.regularization is not None:
            for output in outputs:
                val_loss_reg_mean += output['val_loss_reg']
            val_loss_reg_mean /= len(outputs)
            metrics['val_loss_reg'] = val_loss_reg_mean
            
        if self.zeroparams is not None:
            metrics["zero_params"] = sum([torch.sum(abs(g)<self.zeroparams)  for g in self.model.parameters()])
        self.log_dict(metrics, prog_bar=True,logger=True,on_epoch=True,on_step=False)
        self.logger.log_hyperparams(self.hparams,metrics=metrics)
        return

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self,learning_rate=1e-3):
        if self.optim != "adam":
            optimizer = self.optim
            for g in optimizer.param_groups:
                g["lr"] = learning_rate
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return optimizer


class ProgressBar(pl.callbacks.ProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            desc='Validation ...',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc='Training',
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            ascii=True)
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            desc='Testing',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar

    def init_sanity_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for the validation sanity run. """
        bar = tqdm(
            desc='Validation sanity check',
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            ascii=True)
        return bar


def Classification_report(model):
    with torch.no_grad():
        model.eval()
        model.to(device)
        pred = []
        target = []
        for x, y in model.val_dataloader():
            x = x.to(device)
            pred.append(model(x).cpu().numpy())
            target.append(y.numpy())
    pred = np.concatenate(pred)
    target = np.concatenate(target)
    out = classification_report(target, pred.argmax(axis=1))
    print(out)

class Logger(pl.loggers.TensorBoardLogger):
    def __init__(self, save_dir: str,
                 name: Union[str, None] = 'default',
                 version: Union[int, str, None] = None,
                 log_graph: bool = False,
                 default_hp_metric: bool = True,
                 **kwargs):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, **kwargs)

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
       # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = self._flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)
