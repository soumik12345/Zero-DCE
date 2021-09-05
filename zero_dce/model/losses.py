import tensorflow as tf


def color_constancy_loss(x):
    mean_rgb = tf.reduce_mean(x, [1, 2], keepdims=True)
    mr = mean_rgb[:, :, :, 0]
    mg = mean_rgb[:, :, :, 1]
    mb = mean_rgb[:, :, :, 2] 
    d_rg = tf.pow(mr - mg, 2)
    d_rb = tf.pow(mr - mb, 2)
    d_gb = tf.pow(mb - mg, 2)
    return tf.pow(tf.pow(d_rg, 2) + tf.pow(d_rb, 2) + tf.pow(d_gb, 2), 0.5)


def exposure_loss(x, mean_val=0.6):
    x = tf.reduce_mean(x, 3, keepdims=True)
    mean = tf.keras.layers.AveragePooling2D(
        pool_size=16, strides=16
    )(x)
    return tf.reduce_mean(tf.pow(mean - mean_val, 2))


def illumination_smoothness_loss(x):
    batch_size = tf.shape(x)[0]
    h_x = tf.shape(x)[1]
    w_x = tf.shape(x)[2]
    count_h =  (tf.shape(x)[2] - 1) * tf.shape(x)[3]
    count_w = tf.shape(x)[2] * (tf.shape(x)[3] - 1)
    h_tv = tf.reduce_sum(tf.pow((x[:, 1:, :, :] - x[:, : h_x - 1, :, :]), 2))
    w_tv = tf.reduce_sum(tf.pow((x[:, :, 1:, :] - x[:, :, : w_x - 1, :]), 2))
    batch_size = tf.cast(batch_size, dtype=tf.float32)
    count_h = tf.cast(count_h, dtype=tf.float32)
    count_w = tf.cast(count_w, dtype=tf.float32)
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SpatialConsistancyLoss:

    def __init__(self):

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32)
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32)
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32)
        
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=4)

    
    def __call__(self, original, enhanced):
        
        original_mean = tf.reduce_mean(original, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(enhanced, 3, keepdims=True)
        original_pool = self.pool(original_mean)		
        enhanced_pool = self.pool(enhanced_mean)
        
        d_original_left = tf.nn.conv2d(
            original_pool, self.left_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_original_right = tf.nn.conv2d(
            original_pool, self.right_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_original_down = tf.nn.conv2d(
            original_pool, self.down_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool, self.left_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool, self.right_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool, self.down_kernel,
            strides=[1, 1, 1, 1], padding='SAME'
        )

        d_left = tf.pow(d_original_left - d_enhanced_left, 2)
        d_right = tf.pow(d_original_right - d_enhanced_right, 2)
        d_up = tf.pow(d_original_up - d_enhanced_up, 2)
        d_down = tf.pow(d_original_down - d_enhanced_down, 2)
        return d_left + d_right + d_up + d_down
