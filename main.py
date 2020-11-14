from glob import glob
from zero_dce import (
    download_dataset, init_wandb,
    Trainer, plot_result
)


# Download Dataset
download_dataset()

# Initialize Wandb
init_wandb(
    project_name='zero-dce', experiment_name='lowlight_experiment',
    wandb_api_key='4c77a6750a931c1b13d4d10a0e058725a7487ba9'
)

# Create Trainer
trainer = Trainer()

# Build Dataset
image_files = glob('./Dataset_Part1/*/*.JPG')
trainer.build_dataloader(image_files)

# Build Model
trainer.build_model()

# Train
trainer.train(epochs=200, log_frequency=100)

# Save model
trainer.save_model('model200.pth')

# Inference
for image_file in image_files[:5]:
    image, enhanced = trainer.infer_gpu(image_file)
    plot_result(image, enhanced)
