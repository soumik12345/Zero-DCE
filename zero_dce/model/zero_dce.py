import tensorflow as tf

from .dce_net import DCENet
from .losses import (
    color_constancy_loss, exposure_loss,
    illumination_smoothness_loss, SpatialConsistancyLoss
)


class ZeroDCE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(ZeroDCE, self).__init__(**kwargs)
        self.dce_model = DCENet()

    def compile(self, learning_rate, **kwargs):
        super(ZeroDCE, self).compile(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.spatial_constancy_loss = SpatialConsistancyLoss()
    
    def get_enhanced_image(self, data, output):
        r1 = output[:, :, :, : 3]
        r2 = output[:, :, :, 3: 6]
        r3 = output[:, :, :, 6: 9]
        r4 = output[:, :, :, 9: 12]
        r5 = output[:, :, :, 12: 15]
        r6 = output[:, :, :, 15: 18]
        r7 = output[:, :, :, 18: 21]
        r8 = output[:, :, :, 21: 24]
        
        x = data + r1 * (tf.pow(data, 2) - data)
        x = x + r2 * (tf.pow(x, 2) - x)
        x = x + r3 * (tf.pow(x, 2) - x)
        enhanced_image = x + r4 * (tf.pow(x, 2) - x)
        x = enhanced_image + r5 * (tf.pow(enhanced_image, 2) - enhanced_image)
        x = x + r6 * (tf.pow(x, 2) - x)	
        x = x + r7 * (tf.pow(x, 2) - x)
        enhanced_image = x + r8 * (tf.pow(x, 2) - x)
        
        return enhanced_image
    
    def train_step(self, data):
        
        with tf.GradientTape() as tape:
            output = self.dce_model(data)
            enhanced_image = self.get_enhanced_image(data, output)
            
            loss_illumination = 200 * illumination_smoothness_loss(output)
            loss_spatial_constancy = tf.reduce_mean(
                self.spatial_constancy_loss(enhanced_image, data)
            )
            loss_color_constancy = 5 * tf.reduce_mean(
                color_constancy_loss(enhanced_image)
            )
            loss_exposure = 10 * tf.reduce_mean(
                exposure_loss(enhanced_image)
            )
            total_loss = loss_illumination + loss_spatial_constancy + loss_color_constancy + loss_exposure
        
        gradients = tape.gradient(
            total_loss, self.dce_model.trainable_weights)
        self.optimizer.apply_gradients(zip(
            gradients, self.dce_model.trainable_weights))
        
        return {
            'total_loss': total_loss,
            'illumination_smoothness_loss': loss_illumination,
            'spatial_constancy_loss': loss_spatial_constancy,
            'color_constancy_loss': loss_color_constancy,
            'exposure_loss': loss_exposure
        }
    
    def save_weights(self, filepath, overwrite=True, save_format='tf', options=None):
        self.dce_model.save_weights(
            filepath, overwrite=overwrite,
            save_format=save_format, options=options
        )
