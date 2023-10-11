import tensorflow as tf


def conv_block(input, num_filters):

    x = tf.keras.layers.Conv2D(num_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(input)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    
    return x


def encoder_block(input, num_filters):
    
    x = conv_block(input, num_filters)
    p = tf.keras.layers.MaxPool2D(2)(x)
    p = tf.keras.layers.Dropout(0.3)(p)

    return x, p


def decoder_block(input, skip, num_filters):

    x = tf.keras.layers.Conv2DTranspose(num_filters, 3, 2, padding="same")(input)
    x = tf.keras.layers.concatenate([x, skip])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = conv_block(x, num_filters)

    return x


def construct_unet(input_shape=(512,512,3)):
    # inputs
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encode   
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    bottleneck = conv_block(p4, 1024)

    # Decode
    u6 = decoder_block(bottleneck, f4, 512)
    u7 = decoder_block(u6, f3, 256)
    u8 = decoder_block(u7, f2, 128)
    u9 = decoder_block(u8, f1, 64)

    # outputs
    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u9)

    # If multiclass, use the output below.
    # outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


if __name__ == '__main__':
    model = construct_unet()
    tf.keras.utils.plot_model(model, to_file='model_shape.png', show_shapes=True)
