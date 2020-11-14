import os
import wandb
import gdown
import subprocess
from matplotlib import pyplot as plt


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


def plot_result(image, enhanced):
    fig = plt.figure(figsize=(12, 12))
    fig.add_subplot(1, 2, 1).set_title('Original Image')
    _ = plt.imshow(image)
    fig.add_subplot(1, 2, 2).set_title('Enhanced Image')
    _ = plt.imshow(enhanced)
    plt.show()
