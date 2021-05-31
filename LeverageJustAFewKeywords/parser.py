import os, click
from train import Trainer
from config import hparams
import torch

@click.command()
@click.option('--epochs', default=3)
@click.option('--gpu', default='1', help='-1 is CPU')
@click.option('--lr', default=hparams['lr'])
@click.option('--batch_size', default=hparams['batch_size'])
@click.option('--pretrained', default=hparams['student']['pretrained'])
@click.option('--num_aspects', default=hparams['student']['num_aspect'])
@click.option('--description', default=hparams['description'])
@click.option('--save_dir', default=hparams['save_dir'])
@click.option('--aspect_init_file', default=hparams['aspect_init_file'])
@click.option('--train_file', default=hparams['train_file'])
@click.option('--test_file', default=hparams['test_file'])
@click.option('--general_asp', default=hparams['general_asp'])
@click.option('--maxlen', default=hparams['maxlen'])
def train(epochs, gpu, **kwargs):
    hparams = set_hparams(**kwargs)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu != -1 else None
    click.echo(f'Now GPU: {torch.cuda.get_device_name(0)}')
    trainer = Trainer(hparams, 'cuda' if gpu != -1 else 'cpu')
    trainer.train(epochs)

def set_hparams(**kwargs):
    for key in kwargs.keys():
        if key in hparams.keys():
            hparams[key] = kwargs[key]
        if key in hparams['student'].keys():
            hparams['student'][key] = kwargs[key]
    return hparams


if __name__ == '__main__':
    train()