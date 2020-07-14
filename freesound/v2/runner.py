from v2.config import DefaultConfig
from v2 import utils
from v2.dataloader import stratified_loaders, test_loaders
from v2.model import model_zoo

from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import F1


class Model(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        m = hparams['model']
        model = model_zoo[m['arch']]
        self.model = model(num_classes=m['num_classes'])

        # configure metrics
        self.metrics = {
            'acc': partial(utils.accuracy, topk=(1, 3)),
            'f1': F1(num_classes=m['num_classes'])
        }

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.hparams['train']['optim']['optimizer']['sgd'])
        scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer, **self.hparams['train']['optim']['scheduler']['cosine_annealing'])
        scheduler2 = lr_scheduler.MultiStepLR(optimizer, **self.hparams['train']['optim']['scheduler']['multi_step_lr'])
        return [optimizer], [scheduler1, scheduler2]

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams['train']['mixup'] and np.random.random(1).item() < self.hparams['train']['mixup_p']:
            x, y_a, y_b, lam = utils.mixup_data(x, y, alpha=3)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))

            out = self(x)
            loss = utils.mixup_criterion(F.cross_entropy, out, y_a, y_b, lam)

            out_pred = out.data.softmax(1)
            acc1a, acc3a = self.metrics['acc'](out_pred, y_a.data)
            acc1b, acc3b = self.metrics['acc'](out_pred, y_b.data)
            acc1 = lam * acc1a + (1 - lam) * acc1b
            acc3 = lam * acc3a + (1 - lam) * acc3b

            f1a = self.metrics['f1'](out_pred.argmax(1), y_a.data)
            f1b = self.metrics['f1'](out_pred.argmax(1), y_b.data)
            f1 = lam * f1a + (1 - lam) * f1b
        else:
            out = self(x)
            loss = F.cross_entropy(out, y)

            out_pred = out.data.softmax(1)
            acc1, acc3 = self.metrics['acc'](out_pred, y.data)
            f1 = self.metrics['f1'](out_pred.argmax(1), y.data)

        kwargs = {
            'loss': loss,
            'f1': f1,
            'acc1': acc1,
            'acc3': acc3,
        }

        return {
            **kwargs,
            'log': kwargs
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        out_pred = out.data.softmax(1)
        acc1, acc3 = self.metrics['acc'](out_pred, y.data)
        f1 = self.metrics['f1'](out_pred.argmax(1), y.data)

        kwargs = {
            'val_loss': loss,
            'val_f1': f1,
            'val_acc1': acc1,
            'val_acc3': acc3,
        }

        return {
            **kwargs,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_acc1 = torch.stack([x['val_acc1'] for x in outputs]).mean()
        avg_acc3 = torch.stack([x['val_acc3'] for x in outputs]).mean()

        kwargs = {
            'val_loss': avg_loss,
            'val_f1': avg_f1,
            'val_acc1': avg_acc1,
            'val_acc3': avg_acc3
        }

        return {
            **kwargs,
            'log': kwargs
        }


def train_on_val(conf: DefaultConfig):
    for i, loaders in enumerate(stratified_loaders(conf), 1):
        if i in conf.train.on_folds:
            tb_logger = pl.loggers.TensorBoardLogger(save_dir='tb_logs',
                                                     name=conf.model.arch,
                                                     version=f'fold_{i}_r')
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                filepath=tb_logger.log_dir + "/{epoch:02d}-{val_acc1:.4f}",
                monitor='val_acc1',
                mode='max',
                verbose=False,
                save_last=True)
            early_stop_callback = pl.callbacks.EarlyStopping(min_delta=0.01,
                                                             patience=6,
                                                             verbose=False,
                                                             monitor='val_acc1',
                                                             mode='max')
            ckp = list(Path(tb_logger.log_dir[:-2]).glob('*.ckpt'))[0]

            trainer = pl.Trainer(logger=tb_logger,
                                 early_stop_callback=early_stop_callback,
                                 gpus=1,
                                 precision=16,
                                 checkpoint_callback=checkpoint_callback,
                                 min_epochs=conf.train.retrain_val.min_epochs,
                                 max_epochs=conf.train.retrain_val.max_epochs,
                                 accumulate_grad_batches=conf.train.retrain_val.grad_accum_batch)

            pl.seed_everything(conf.train.seed)
            model = Model.load_from_checkpoint(checkpoint_path=str(ckp))
            trainer.fit(model, train_dataloader=loaders['val'], val_dataloaders=loaders['train'])

            ckp = list(Path(tb_logger.log_dir).glob('*.ckpt'))[0]
            model = Model.load_from_checkpoint(checkpoint_path=str(ckp))
            model.to(conf.train.device)
            model.freeze()

            print(f'Test preds for fold_{i}_r ...')
            preds = []
            for x in loaders['test']:
                y_preds = model(x.to(conf.train.device))
                y_preds = y_preds.data.cpu().softmax(1).numpy()
                preds.append(y_preds)
            preds = np.concatenate(preds, axis=0)
            np.save(Path(tb_logger.log_dir).joinpath('preds.npy'), preds)

            ## Make several preds for averaging
            for j, loader in enumerate(test_loaders(conf, n=5)):
                print(f'Test preds for fold_{i}_v{j} ...')
                preds = []
                for x in loader:
                    y_preds = model(x.to(conf.train.device))
                    y_preds = y_preds.data.cpu().softmax(1).numpy()
                    preds.append(y_preds)
                preds = np.concatenate(preds, axis=0)
                np.save(Path(tb_logger.log_dir).joinpath(f'preds_v{j}.npy'), preds)


if __name__ == '__main__':
    conf = DefaultConfig()
    for i, loaders in enumerate(stratified_loaders(conf), 1):
        if i in conf.train.on_folds:
            tb_logger = pl.loggers.TensorBoardLogger(save_dir='tb_logs',
                                                     name=conf.model.arch,
                                                     version=f'fold_{i}')
            checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{val_acc1:.4f}",
                                                               monitor='val_acc1',
                                                               mode='max',
                                                               verbose=False)
            early_stop_callback = pl.callbacks.EarlyStopping(min_delta=0.01,
                                                             patience=12,
                                                             verbose=False,
                                                             monitor='val_acc1',
                                                             mode='max')
            trainer = pl.Trainer(logger=tb_logger,
                                 early_stop_callback=early_stop_callback,
                                 gpus=1,
                                 precision=16,
                                 checkpoint_callback=checkpoint_callback,
                                 min_epochs=conf.train.min_epochs,
                                 max_epochs=conf.train.max_epochs,
                                 accumulate_grad_batches=conf.train.grad_accum_batch)
            pl.seed_everything(conf.train.seed)
            model = Model(hparams=conf.to_dict())
            trainer.fit(model, train_dataloader=loaders['train'], val_dataloaders=loaders['val'])

            ckp = list(Path(tb_logger.log_dir).glob('*.ckpt'))[0]
            model = Model.load_from_checkpoint(checkpoint_path=str(ckp))
            model.to(conf.train.device)
            model.freeze()

            print(f'Test preds for fold_{i} ...')
            preds = []
            for x in loaders['test']:
                y_preds = model(x.to(conf.train.device))
                y_preds = y_preds.data.cpu().softmax(1).numpy()
                preds.append(y_preds)
            preds = np.concatenate(preds, axis=0)
            np.save(Path(tb_logger.log_dir).joinpath('preds.npy'), preds)

            ## Make several preds for averaging
            for j, loader in enumerate(test_loaders(conf, n=5)):
                print(f'Test preds for fold_{i}_v{j} ...')
                preds = []
                for x in loader:
                    y_preds = model(x.to(conf.train.device))
                    y_preds = y_preds.data.cpu().softmax(1).numpy()
                    preds.append(y_preds)
                preds = np.concatenate(preds, axis=0)
                np.save(Path(tb_logger.log_dir).joinpath(f'preds_v{j}.npy'), preds)

    ## Retrain on validation
    conf = DefaultConfig()
    train_on_val(conf)