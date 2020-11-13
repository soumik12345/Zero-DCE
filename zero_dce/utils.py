import os
import wandb
import gdown
import subprocess


def download_dataset():
    print('Downloading dataset...')
    gdown.download(
        'https://drive.google.com/uc?id=1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN',
        'Dataset_Part1.rar', quiet=False
    )
    print('Unpacking Dataset')
    subprocess.run('unrar x Dataset_Part1.rar'.split(' '))
    print('Done!!!')


def init_wandb(project_name, experiment_name, wandb_api_key):
    if project_name is not None and experiment_name is not None:
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(project=project_name, name=experiment_name)
