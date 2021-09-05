import os
import click
from glob import glob
import tensorflow as tf
from datetime import datetime

from zero_dce.dataloader import LowLightDataLoader
from zero_dce.model import ZeroDCE, zero_dce


@click.command()
@click.option('--dataset_path', default='./data/Low', help='Low-light dataset path')
@click.option('--image_size', default=512, help='Image size')
@click.option('--batch_size', default=4, help='Batch size')
@click.option('--learning_rate', default=1e-4, help='Learning rate')
@click.option('--epochs', default=200, help='Epochs')
def main(dataset_path, image_size, batch_size, learning_rate, epochs):
    
    dataloader = LowLightDataLoader(
        low_light_images=glob(
            os.path.join(dataset_path, '*.png')
        ), image_size=image_size
    )
    dataset = dataloader.get_dataset(batch_size=batch_size)
    
    zero_dce_model = ZeroDCE()
    zero_dce_model.compile(learning_rate=learning_rate)
    
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            os.path.join(
                'logs', datetime.now().strftime('%Y%m%d-%H%M%S')
            ), histogram_freq=1
        )
    ]
    zero_dce_model.fit(dataset, epochs=epochs, callbacks=callbacks)
    zero_dce.save_weights('zero-dce-{}'.format(epochs), save_format='tf')


if __name__ == '__main__':
    main()
