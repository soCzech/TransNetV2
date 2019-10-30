import gin
import h5py
import numpy as np
import tensorflow as tf

from models import ResNet18, ResNetBlock


@gin.configurable(blacklist=["name"])
class TransNetV2(tf.keras.Model):

    def __init__(self, F=16, L=3, S=2, D=256,
                 use_resnet_features=False,
                 use_many_hot_targets=False,
                 use_frame_similarity=False,
                 use_mean_pooling=False,
                 name="TransNet"):
        super(TransNetV2, self).__init__(name=name)

        self.resnet_layers = ResNetFeatures() if use_resnet_features else (lambda x, training=False: x / 255.)
        self.blocks = [StackedDDCNNV2(n_blocks=S, filters=F * 2**i, name="SDDCNN_{:d}".format(i + 1)) for i in range(L)]
        self.fc1 = tf.keras.layers.Dense(D, activation=tf.nn.relu)
        self.cls_layer1 = tf.keras.layers.Dense(1, activation=None)
        self.cls_layer2 = tf.keras.layers.Dense(1, activation=None) if use_many_hot_targets else None
        self.frame_sim_layer = FrameSimilarity() if use_frame_similarity else None
        self.use_mean_pooling = use_mean_pooling

    def call(self, inputs, training=False):
        x = inputs
        x = self.resnet_layers(x, training=training)

        block_features = []
        for block in self.blocks:
            x = block(x, training=training)
            block_features.append(x)

        if self.use_mean_pooling:
            x = tf.math.reduce_mean(x, axis=[2, 3])
        else:
            shape = [tf.shape(x)[0], tf.shape(x)[1], np.prod(x.get_shape().as_list()[2:])]
            x = tf.reshape(x, shape=shape, name="flatten_3d")

        if self.frame_sim_layer is not None:
            x = tf.concat([self.frame_sim_layer(block_features), x], 2)

        x = self.fc1(x)
        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            many_hot = self.cls_layer2(x)
            return one_hot, many_hot

        return one_hot


@gin.configurable(whitelist=["shortcut"])
class StackedDDCNNV2(tf.keras.layers.Layer):

    def __init__(self, n_blocks, filters, shortcut=False, name="StackedDDCNN"):
        super(StackedDDCNNV2, self).__init__(name=name)
        self.shortcut = None
        if shortcut:
            self.shortcut = tf.keras.layers.Conv3D(filters * 4, kernel_size=1, dilation_rate=1, padding="SAME",
                                                   activation=None, use_bias=True, name="shortcut")

        self.blocks = [DilatedDCNNV2(filters, activation=tf.nn.relu if i != n_blocks else None,
                                     name="DDCNN_{:d}".format(i)) for i in range(1, n_blocks + 1)]
        self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))

    def call(self, inputs, training=False):
        x = inputs
        for block in self.blocks:
            x = block(x, training=training)

        if self.shortcut is not None:
            x += self.shortcut(inputs)
        x = tf.nn.relu(x)

        x = self.max_pool(x)
        return x


@gin.configurable(whitelist=["batch_norm"])
class DilatedDCNNV2(tf.keras.layers.Layer):

    def __init__(self, filters, batch_norm=False, activation=None, name="DilatedDCNN"):
        super(DilatedDCNNV2, self).__init__(name=name)

        self.conv1 = Conv3DConfigurable(filters, 1, use_bias=not batch_norm, name="Conv3D_1")
        self.conv2 = Conv3DConfigurable(filters, 2, use_bias=not batch_norm, name="Conv3D_2")
        self.conv3 = Conv3DConfigurable(filters, 4, use_bias=not batch_norm, name="Conv3D_4")
        self.conv4 = Conv3DConfigurable(filters, 8, use_bias=not batch_norm, name="Conv3D_8")

        self.batch_norm = tf.keras.layers.BatchNormalization(name="bn") if batch_norm else None
        self.activation = activation

    def call(self, inputs, training=False):
        inputs = tf.identity(inputs)
        conv1 = self.conv1(inputs, training=training)
        conv2 = self.conv2(inputs, training=training)
        conv3 = self.conv3(inputs, training=training)
        conv4 = self.conv4(inputs, training=training)
        x = tf.concat([conv1, conv2, conv3, conv4], axis=4)

        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        if self.activation is not None:
            x = self.activation(x)
        return x


@gin.configurable(whitelist=["separable"])
class Conv3DConfigurable(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 dilation_rate,
                 separable=False,
                 use_bias=True,
                 name="Conv3D"):
        super(Conv3DConfigurable, self).__init__(name=name)

        if separable:
            conv1 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 1, 1), dilation_rate=(dilation_rate, 1, 1),
                                           padding="SAME", activation=None, use_bias=False, name="conv_temporal")
            conv2 = tf.keras.layers.Conv3D(filters, kernel_size=(1, 3, 3), dilation_rate=(1, 1, 1),
                                           padding="SAME", activation=None, use_bias=use_bias,
                                           name="conv_spatial")
            self.layers = [conv1, conv2]
        else:
            conv = tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                          padding="SAME", activation=None, use_bias=use_bias, name="conv")
            self.layers = [conv]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


@gin.configurable(whitelist=["trainable"])
class ResNetFeatures(tf.keras.layers.Layer):

    def __init__(self, trainable=False, name="ResNetFeatures"):
        super(ResNetFeatures, self).__init__(trainable=trainable, name=name)

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                                            padding="SAME", use_bias=False, name="conv1")
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")
        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="SAME")

        self.layer2a = ResNetBlock(64, name="Block2a")
        self.layer2b = ResNetBlock(64, name="Block2b")

        self.mean = tf.constant(ResNet18.MEAN)
        self.std = tf.constant(ResNet18.STD)

    def call(self, inputs, training=False):
        training = training if self.trainable else False
        shape = tf.shape(inputs)

        x = tf.reshape(inputs, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
        x = (x - self.mean) / self.std

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.layer2a(x, training=training)
        x = self.layer2b(x, training=training)

        new_shape = tf.shape(x)
        x = tf.reshape(x, [shape[0], shape[1], new_shape[1], new_shape[2], new_shape[3]])
        return x

    def restore_me(self, checkpoint):
        with h5py.File(checkpoint, "r") as f:
            for v in self.variables:
                name = v.name.split("/")[2:]
                if name[0].startswith("Block"):
                    name = name[:1] + name
                else:
                    name = name[:len(name) - 1] + name
                name = "/".join(name)
                v.assign(f[name][:])


@gin.configurable(whitelist=["similarity_dim", "lookup_window", "output_dim"])
class FrameSimilarity(tf.keras.layers.Layer):

    def __init__(self,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 name="FrameSimilarity"):
        super(FrameSimilarity, self).__init__(name=name)

        self.projection = tf.keras.layers.Dense(similarity_dim, use_bias=False, activation=None)
        self.fc = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def call(self, inputs):
        x = tf.concat([
            tf.math.reduce_mean(x, axis=[2, 3]) for x in inputs
        ], axis=2)

        x = self.projection(x)
        x = tf.nn.l2_normalize(x, axis=2)

        batch_size, time_window = tf.shape(x)[0], tf.shape(x)[1]
        similarities = tf.matmul(x, x, transpose_b=True)  # [batch_size, time_window, time_window]
        similarities_padded = tf.pad(similarities, [[0, 0], [0, 0], [(self.lookup_window - 1) // 2] * 2])

        batch_indices = tf.tile(
            tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, time_window, self.lookup_window]
        )
        time_indices = tf.tile(
            tf.reshape(tf.range(time_window), [1, time_window, 1]), [batch_size, 1, self.lookup_window]
        )
        lookup_indices = tf.tile(
            tf.reshape(tf.range(self.lookup_window), [1, 1, self.lookup_window]), [batch_size, time_window, 1]
        ) + time_indices

        indices = tf.stack([batch_indices, time_indices, lookup_indices], -1)

        similarities = tf.gather_nd(similarities_padded, indices)
        return self.fc(similarities)
