import os
import gc
import torch
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


def pretty_size(size):
    """Pretty prints a torch.Size object"""
    assert (isinstance(size, torch.Size))
    return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
    """Prints a list of the Tensors being tracked by the garbage collector."""
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                          " GPU" if obj.is_cuda else "",
                                          " pinned" if obj.is_pinned else "",
                                          pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                                   type(obj.data).__name__,
                                                   " GPU" if obj.is_cuda else "",
                                                   " pinned" if obj.data.is_pinned else "",
                                                   " grad" if obj.requires_grad else "",
                                                   " volatile" if obj.volatile else "",
                                                   pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            print(e)
    print("Total size:", total_size)
