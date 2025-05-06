import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# I reference Microsoft CNTK Official ResNet code. See following links.
# https://github.com/microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/resnet_models.py
# https://github.com/microsoft/CNTK/blob/master/Examples/Image/Classification/ResNet/Python/TrainResNet_ImageNet_Distributed.py

def conv_bn(x, num_filters, kernel_size, strides=(1,1), 
            init="he_normal", gamma_init="ones"):
    
    c = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer=init,
        use_bias=False
    )(x)

    return layers.BatchNormalization(gamma_initializer=gamma_init)(c)

def conv_bn_relu(x, num_filters, kernel_size, strides=(1,1), 
                 init="he_normal", gamma_init="ones"):
    
    c = conv_bn(
        x, num_filters, kernel_size, strides, init, gamma_init
        )

    return layers.Activation("relu")(c)

def stack_blocks(x, block_func, count, **kwargs):

    for _ in range(count):
        x = block_func(x, **kwargs)
    
    return x

def bottleneck_block(x, num_filters, inter_num_filters):

    c1 = conv_bn_relu(x, inter_num_filters, (1, 1))
    c2 = conv_bn_relu(c1, inter_num_filters, (3, 3))
    c3 = conv_bn(c2, num_filters, (1, 1), gamma_init="zeros")

    p = layers.Add()([c3, x])

    return layers.Activation("relu")(p)

def bottleneck_block_inc(x, num_filters, inter_num_filters, 
                         stride1x1=(1, 1), stride2x2=(2, 2)):
    
    c1 = conv_bn_relu(x, inter_num_filters, (1, 1), strides=stride1x1)
    
    #Resnet152 use stride2x2 instead of stride3x3
    c2 = conv_bn_relu(c1, inter_num_filters, (3, 3), strides=stride2x2)

    c3 = conv_bn(c2, num_filters, (1, 1), gamma_init="zeros")

    stride = np.multiply(stride1x1, stride2x2)

    shortcut = conv_bn(x, num_filters, (1, 1), strides=stride)

    p = layers.Add()([c3 + shortcut])

    return layers.Activation("relu")(p)

def resnet_bottleneck_model(x, 
                            num_filters_list, block_repeat_list, 
                            include_final_pool=False):
    
    assert len(num_filters_list[2:]) == len(block_repeat_list), "num_filters_list[2:] and bloack_repeat_list should have same length"

    x = conv_bn_relu(x, num_filters_list[0], (7, 7), strides=(2, 2))
    x = layers.MaxPool2D(
        pool_size=(3,3), strides=(2, 2), padding="same")(x)
    
    # stack resnet bottleneck block via `block_repeat_list`
    for i in range(len(block_repeat_list)):
        # apply downsample from sencond block
        stride = (1, 1) if i == 0 else (2, 2)
        x = bottleneck_block_inc(
            x, num_filters=num_filters_list[i + 2], 
            inter_num_filters=num_filters_list[i + 2] // 4, 
            stride2x2=stride)
        
        x = stack_blocks(
            x, block_func=bottleneck_block,
            count=block_repeat_list[i],
            num_filters=num_filters_list[i + 2],
            inter_num_filters=num_filters_list[i + 2] // 4
        )

    
    if include_final_pool:
        x = layers.AveragePooling2D(
            pool_size=(7, 7), name="final_average_pooling"
        )(x)

    return x