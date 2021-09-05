from typing import List
import tensorflow as tf


class LowLightDataLoader:

    def __init__(self, low_light_images: List[str], image_size: int = 512):
        self.low_light_images = low_light_images
        self.image_size = image_size

    def load_data(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(
            images=image, size=[
                self.image_size,
                self.image_size
            ]
        )
        image = image / 255.0
        return image

    def get_dataset(self, batch_size: int = 4):
        dataset = tf.data.Dataset.from_tensor_slices(self.low_light_images)
        dataset = dataset.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset
