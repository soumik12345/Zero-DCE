import tensorflow as tf


class DCENet(tf.keras.Model):

    def __init__(self):
        super(DCENet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.conv2 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.conv3 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.conv4 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.conv5 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.conv6 = tf.keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1),
            activation='relu', padding='same'
        )
        self.output_conv = tf.keras.layers.Conv2D(
            24, (3, 3), strides=(1, 1),
            activation='tanh', padding='same'
        )
        self.concatenate = tf.keras.layers.Concatenate(axis=-1)
    
    def call(self, image):
        x1 = self.conv1(image)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(self.concatenate([x4, x3]))
        x6 = self.conv6(self.concatenate([x5, x2]))
        return self.output_conv(self.concatenate([x6, x1]))
