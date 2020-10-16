import gin
import numpy as np
import tensorflow as tf


class OriginalTransNet(tf.keras.Model):

    def __init__(self, F=16, L=3, S=2, D=256, name="TransNet"):
        super(OriginalTransNet, self).__init__(name=name)

        self.blocks = [StackedDDCNN(n_blocks=S, filters=F * 2 ** i, name="SDDCNN_{:d}".format(i + 1)) for i in range(L)]
        self.fc1 = tf.keras.layers.Dense(D, activation=tf.nn.relu)
        self.fc2 = tf.keras.layers.Dense(2, activation=None)

    def call(self, inputs, training=False):
        x = inputs / 255.
        for block in self.blocks:
            x = block(x)

        shape = [tf.shape(x)[0], tf.shape(x)[1], np.prod(x.get_shape().as_list()[2:])]
        x = tf.reshape(x, shape=shape, name="flatten_3d")

        x = self.fc1(x)
        x = self.fc2(x)
        return x


class StackedDDCNN(tf.keras.layers.Layer):

    def __init__(self, n_blocks, filters, name="StackedDDCNN"):
        super(StackedDDCNN, self).__init__(name=name)
        self.blocks = [DilatedDCNN(filters, name="DDCNN_{:d}".format(i)) for i in range(1, n_blocks + 1)]
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)

        x = self.max_pool(x)
        return x


class DilatedDCNN(tf.keras.layers.Layer):

    def __init__(self, filters, name="DilatedDCNN"):
        super(DilatedDCNN, self).__init__(name=name)

        self.conv1 = self._conv3d(filters, 1, name="Conv3D_1")
        self.conv2 = self._conv3d(filters, 2, name="Conv3D_2")
        self.conv3 = self._conv3d(filters, 4, name="Conv3D_4")
        self.conv4 = self._conv3d(filters, 8, name="Conv3D_8")

    @staticmethod
    def _conv3d(filters, dilation_rate, name="Conv3D"):
        return tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                      padding="SAME", activation=tf.nn.relu, use_bias=True, name=name)

    def call(self, inputs):
        inputs = tf.identity(inputs)
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(inputs)
        conv4 = self.conv4(inputs)
        x = tf.concat([conv1, conv2, conv3, conv4], axis=4)
        return x


class ResNet18(tf.keras.Model):

    MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape([1, 1, 1, 3]) * 255
    STD = np.array([0.229, 0.224, 0.225], np.float32).reshape([1, 1, 1, 3]) * 255

    def __init__(self, name="ResNet18"):
        super(ResNet18, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                                            padding="SAME", use_bias=False, name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")

        self.layer2a = ResNetBlock(64, name="Block2a")
        self.layer2b = ResNetBlock(64, name="Block2b")

        self.layer3a = ResNetBlock(128, strides=(2, 2), project=True, name="Block3a")
        self.layer3b = ResNetBlock(128, name="Block3b")

        self.layer4a = ResNetBlock(256, strides=(2, 2), project=True, name="Block4a")
        self.layer4b = ResNetBlock(256, name="Block4b")

        self.layer5a = ResNetBlock(512, strides=(2, 2), project=True, name="Block5a")
        self.layer5b = ResNetBlock(512, name="Block5b")

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7))

        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(1000)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.layer2a(x, training=training)
        x = self.layer2b(x, training=training)

        x = self.layer3a(x, training=training)
        x = self.layer3b(x, training=training)

        x = self.layer4a(x, training=training)
        x = self.layer4b(x, training=training)

        x = self.layer5a(x, training=training)
        x = self.layer5b(x, training=training)

        x = self.avg_pool(x)
        return self.fc(self.flatten(x))

    @staticmethod
    def preprocess(inputs):
        assert inputs.dtype == np.uint8 or inputs.dtype == tf.uint8
        if len(inputs.shape) == 3:
            inputs = inputs[tf.newaxis]
        assert inputs.shape[1:] == (224, 224, 3)

        mean = tf.constant(ResNet18.MEAN)
        std = tf.constant(ResNet18.STD)

        x = tf.cast(inputs, tf.float32)
        return (x - mean) / std


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), project=False, name="ResNetBlock"):
        super(ResNetBlock, self).__init__(name=name)

        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides,
                                            padding="SAME", use_bias=False, name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="SAME", use_bias=False, name="conv2")
        self.bn2 = tf.keras.layers.BatchNormalization(gamma_initializer=tf.zeros_initializer(), name="conv2/bn")

        self.project = project
        if self.project:
            self.conv_shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides,
                                                        use_bias=False, name="conv_shortcut")
            self.bn_shortcut = tf.keras.layers.BatchNormalization(name="conv_shortcut/bn")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        shortcut = inputs
        if self.project:
            shortcut = self.conv_shortcut(shortcut)
            shortcut = self.bn_shortcut(shortcut, training=training)
        x += shortcut

        return tf.nn.relu(x)


@gin.configurable(blacklist=["name"])
class C3DConvolutions(tf.keras.Model):
    # C3D model for UCF101
    # https://github.com/tqvinhcs/C3D-tensorflow/blob/master/m_c3d.py#L63

    def __init__(self, weights=None, restore_from=None, name="C3DConvolutions"):
        super(C3DConvolutions, self).__init__(name=name)
        if restore_from is not None:
            weights = self.get_weights(restore_from)
        elif weights is None:
            weights = [None] * 16

        def conv(filters, kernel_weights, bias_weights):
            return tf.keras.layers.Conv3D(filters, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu,
                                          kernel_initializer=tf.constant_initializer(kernel_weights) \
                                              if kernel_weights is not None else "glorot_uniform",
                                          bias_initializer=tf.constant_initializer(bias_weights) \
                                              if bias_weights is not None else "zeros")

        self.conv_layers = [
            conv(f, ker_init, bias_init) for f, ker_init, bias_init in [
                (64, weights[0], weights[1]),
                (128, weights[2], weights[3]),
                (256, weights[4], weights[5]),
                (256, weights[6], weights[7]),
                (512, weights[8], weights[9]),
                (512, weights[10], weights[11]),
                (512, weights[12], weights[13]),
                (512, weights[14], weights[15])
            ]
        ]
        self.max_pooling = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="SAME")

    def call(self, inputs, training=False):
        x = inputs - 96.6
        print(x.shape)
        x = self.conv_layers[0](x)
        x = self.max_pooling(x)
        print(x.shape)
        x = self.conv_layers[1](x)
        x = self.max_pooling(x)
        print(x.shape)
        x = self.conv_layers[2](x)
        x = self.conv_layers[3](x)
        x = self.max_pooling(x)
        print(x.shape)
        x = self.conv_layers[4](x)
        x = self.conv_layers[5](x)
        x = self.max_pooling(x)
        print(x.shape)
        x = self.conv_layers[6](x)
        x = self.conv_layers[7](x)
        x = self.max_pooling(x)
        print(x.shape)
        return x

    @staticmethod
    def get_weights(filename):
        import scipy.io as sio
        return sio.loadmat(filename, squeeze_me=True)['weights']


@gin.configurable(blacklist=["name"])
class C3DNet(tf.keras.Model):

    def __init__(self, D=256, name="C3DNet"):
        super(C3DNet, self).__init__(name=name)
        self.convs = C3DConvolutions()
        self.fc1 = tf.keras.layers.Dense(D, activation=tf.nn.relu)
        self.cls_layer1 = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        x = self.convs(inputs, training=training)
        x = tf.math.reduce_mean(x, axis=[2, 3])
        x = self.fc1(x)
        x = self.cls_layer1(x)

        return x
