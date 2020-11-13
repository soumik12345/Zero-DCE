from glob import glob
from zero_dce import download_dataset, init_wandb, Trainer


download_dataset()
init_wandb(
    project_name='zero-dce', experiment_name='lowlight_experiment',
    wandb_api_key='4c77a6750a931c1b13d4d10a0e058725a7487ba9'
)
trainer = Trainer()
image_files = glob('./Dataset_Part1/*/*.JPG')
trainer.build_dataloader(image_files)
trainer.build_model()
trainer.train(epochs=200, log_frequency=100)
