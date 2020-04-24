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
                 use_convex_comb_reg=False,
                 dropout_rate=None,
                 use_resnet_like_top=False,
                 frame_similarity_on_last_layer=False,
                 use_color_histograms=False,
                 name="TransNet"):
        super(TransNetV2, self).__init__(name=name)

        self.resnet_layers = ResNetFeatures() if use_resnet_features else (lambda x, training=False: x / 255.)
        self.blocks = [StackedDDCNNV2(n_blocks=S, filters=F, stochastic_depth_drop_prob=0., name="SDDCNN_1")]
        self.blocks += [StackedDDCNNV2(n_blocks=S, filters=F * 2**i, name="SDDCNN_{:d}".format(i + 1)) for i in range(1, L)]
        self.fc1 = tf.keras.layers.Dense(D, activation=tf.nn.relu)
        self.cls_layer1 = tf.keras.layers.Dense(1, activation=None)
        self.cls_layer2 = tf.keras.layers.Dense(1, activation=None) if use_many_hot_targets else None
        self.frame_sim_layer = FrameSimilarity() if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms() if use_color_histograms else None
        self.use_mean_pooling = use_mean_pooling
        self.convex_comb_reg = ConvexCombinationRegularization() if use_convex_comb_reg else None
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate is not None else None

        self.frame_similarity_on_last_layer = frame_similarity_on_last_layer
        self.resnet_like_top = use_resnet_like_top
        if self.resnet_like_top:
            self.resnet_like_top_conv = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 7, 7), strides=(1, 2, 2),
                                                               padding="SAME", use_bias=False,
                                                               name="resnet_like_top/conv")
            self.resnet_like_top_bn = tf.keras.layers.BatchNormalization(name="resnet_like_top/bn")
            self.resnet_like_top_max_pool = tf.keras.layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2),
                                                                         padding="SAME")

    def call(self, inputs, training=False):
        out_dict = {}

        x = inputs
        x = self.resnet_layers(x, training=training)

        if self.resnet_like_top:
            x = self.resnet_like_top_conv(x)
            x = self.resnet_like_top_bn(x)
            x = self.resnet_like_top_max_pool(x)

        block_features = []
        for block in self.blocks:
            x = block(x, training=training)
            block_features.append(x)

        if self.convex_comb_reg is not None:
            out_dict["alphas"], out_dict["comb_reg_loss"] = self.convex_comb_reg(inputs, x)

        if self.use_mean_pooling:
            x = tf.math.reduce_mean(x, axis=[2, 3])
        else:
            shape = [tf.shape(x)[0], tf.shape(x)[1], np.prod(x.get_shape().as_list()[2:])]
            x = tf.reshape(x, shape=shape, name="flatten_3d")

        if self.frame_sim_layer is not None and not self.frame_similarity_on_last_layer:
            x = tf.concat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = tf.concat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)

        if self.frame_sim_layer is not None and self.frame_similarity_on_last_layer:
            x = tf.concat([self.frame_sim_layer(block_features), x], 2)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            out_dict["many_hot"] = self.cls_layer2(x)

        if len(out_dict) > 0:
            return one_hot, out_dict
        return one_hot


@gin.configurable(whitelist=["shortcut", "use_octave_conv", "pool_type", "stochastic_depth_drop_prob"])
class StackedDDCNNV2(tf.keras.layers.Layer):

    def __init__(self, n_blocks, filters, shortcut=False, use_octave_conv=False, pool_type="max",
                 stochastic_depth_drop_prob=0., name="StackedDDCNN"):
        super(StackedDDCNNV2, self).__init__(name=name)
        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        # self.shortcut = None
        # if shortcut:
        #     self.shortcut = tf.keras.layers.Conv3D(filters * 4, kernel_size=1, dilation_rate=1, padding="SAME",
        #                                            activation=None, use_bias=True, name="shortcut")

        self.blocks = [DilatedDCNNV2(filters, octave_conv=use_octave_conv,
                                     activation=tf.nn.relu if i != n_blocks else None,
                                     name="DDCNN_{:d}".format(i)) for i in range(1, n_blocks + 1)]
        self.pool = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2)) if pool_type == "max" else \
            tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2))
        self.octave = use_octave_conv
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def call(self, inputs, training=False):
        x = inputs
        shortcut = None

        if self.octave:
            x = [self.pool(x), x]
        for block in self.blocks:
            x = block(x, training=training)
            if shortcut is None:
                shortcut = x
        if self.octave:
            x = tf.concat([x[0], self.pool(x[1])], -1)

        x = tf.nn.relu(x)

        if self.shortcut is not None:
            # shortcut = self.shortcut(inputs)
            if self.stochastic_depth_drop_prob != 0.:
                if training:
                    x = tf.cond(tf.random.uniform([]) < self.stochastic_depth_drop_prob,
                                lambda: shortcut, lambda: x + shortcut)
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut

        if not self.octave:
            x = self.pool(x)
        return x


@gin.configurable(whitelist=["batch_norm"])
class DilatedDCNNV2(tf.keras.layers.Layer):

    def __init__(self, filters, batch_norm=False, activation=None, octave_conv=False, name="DilatedDCNN"):
        super(DilatedDCNNV2, self).__init__(name=name)
        assert not (octave_conv and batch_norm)

        self.conv1 = Conv3DConfigurable(filters, 1, use_bias=not batch_norm, octave=octave_conv, name="Conv3D_1")
        self.conv2 = Conv3DConfigurable(filters, 2, use_bias=not batch_norm, octave=octave_conv, name="Conv3D_2")
        self.conv3 = Conv3DConfigurable(filters, 4, use_bias=not batch_norm, octave=octave_conv, name="Conv3D_4")
        self.conv4 = Conv3DConfigurable(filters, 8, use_bias=not batch_norm, octave=octave_conv, name="Conv3D_8")
        self.octave = octave_conv

        self.batch_norm = tf.keras.layers.BatchNormalization(name="bn") if batch_norm else None
        self.activation = activation

    def call(self, inputs, training=False):
        conv1 = self.conv1(inputs, training=training)
        conv2 = self.conv2(inputs, training=training)
        conv3 = self.conv3(inputs, training=training)
        conv4 = self.conv4(inputs, training=training)

        if self.octave:
            x = [tf.concat([conv1[0], conv2[0], conv3[0], conv4[0]], axis=4),
                 tf.concat([conv1[1], conv2[1], conv3[1], conv4[1]], axis=4)]
        else:
            x = tf.concat([conv1, conv2, conv3, conv4], axis=4)

        if self.batch_norm is not None:
            x = self.batch_norm(x, training=training)

        if self.activation is not None:
            if self.octave:
                x = [self.activation(x[0]), self.activation(x[1])]
            else:
                x = self.activation(x)
        return x


@gin.configurable(whitelist=["separable", "kernel_initializer"])
class Conv3DConfigurable(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 dilation_rate,
                 separable=False,
                 octave=False,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 name="Conv3D"):
        super(Conv3DConfigurable, self).__init__(name=name)
        assert not (separable and octave)

        if separable:
            # (2+1)D convolution https://arxiv.org/pdf/1711.11248.pdf
            conv1 = tf.keras.layers.Conv3D(2 * filters, kernel_size=(1, 3, 3), dilation_rate=(1, 1, 1),
                                           padding="SAME", activation=None, use_bias=False,
                                           name="conv_spatial", kernel_initializer=kernel_initializer)
            conv2 = tf.keras.layers.Conv3D(filters, kernel_size=(3, 1, 1), dilation_rate=(dilation_rate, 1, 1),
                                           padding="SAME", activation=None, use_bias=use_bias, name="conv_temporal",
                                           kernel_initializer=kernel_initializer)
            self.layers = [conv1, conv2]
        elif octave:
            conv = OctConv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1), use_bias=use_bias,
                             kernel_initializer=kernel_initializer)
            self.layers = [conv]
        else:
            conv = tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
                                          padding="SAME", activation=None, use_bias=use_bias, name="conv",
                                          kernel_initializer=kernel_initializer)
            self.layers = [conv]

    def call(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


@gin.configurable(whitelist=["alpha"])
class OctConv3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, dilation_rate=(1, 1, 1), alpha=0.25,
                 use_bias=True, kernel_initializer="glorot_uniform", name="OctConv3D"):
        super(OctConv3D, self).__init__(name=name)

        self.low_channels = int(filters * alpha)
        self.high_channels = filters - self.low_channels

        self.high_to_high = tf.keras.layers.Conv3D(self.high_channels, kernel_size=kernel_size, activation=None,
                                                   dilation_rate=dilation_rate, padding="SAME",
                                                   use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                   name="high_to_high")
        self.high_to_low = tf.keras.layers.Conv3D(self.low_channels, kernel_size=kernel_size, activation=None,
                                                  dilation_rate=dilation_rate, padding="SAME",
                                                  use_bias=False, kernel_initializer=kernel_initializer,
                                                  name="high_to_low")
        self.low_to_high = tf.keras.layers.Conv3D(self.high_channels, kernel_size=kernel_size, activation=None,
                                                  dilation_rate=dilation_rate, padding="SAME",
                                                  use_bias=False, kernel_initializer=kernel_initializer,
                                                  name="low_to_high")
        self.low_to_low = tf.keras.layers.Conv3D(self.low_channels, kernel_size=kernel_size, activation=None,
                                                 dilation_rate=dilation_rate, padding="SAME",
                                                 use_bias=use_bias, kernel_initializer=kernel_initializer,
                                                 name="low_to_low")
        self.upsampler = tf.keras.layers.UpSampling3D(size=(1, 2, 2))
        self.downsampler = tf.keras.layers.AveragePooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding="SAME")

    @staticmethod
    def pad_to(tensor, target_shape):
        shape = tf.shape(tensor)
        padding = [[0, tar - curr] for curr, tar in zip(shape, target_shape)]
        return tf.pad(tensor, padding, "CONSTANT")

    @staticmethod
    def crop_to(tensor, target_width, target_height):
        return tensor[:, :, :target_height, :target_width]

    def call(self, inputs):
        low_inputs, high_inputs = inputs

        high_to_high = self.high_to_high(high_inputs)
        high_to_low = self.high_to_low(self.downsampler(high_inputs))

        low_to_high = self.upsampler(self.low_to_high(low_inputs))
        low_to_low = self.low_to_low(low_inputs)

        high_output = high_to_high[:, :, :tf.shape(low_to_high)[2], :tf.shape(low_to_high)[3]] + low_to_high
        low_output = low_to_low + high_to_low[:, :, :tf.shape(low_to_low)[2], :tf.shape(low_to_low)[3]]

        # print("OctConv3D:", low_inputs.shape, "->", low_output.shape, "|", high_inputs.shape, "->", high_output.shape)
        return low_output, high_output


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


@gin.configurable(whitelist=["similarity_dim", "lookup_window", "output_dim", "stop_gradient", "use_bias"])
class FrameSimilarity(tf.keras.layers.Layer):

    def __init__(self,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,
                 use_bias=False,
                 name="FrameSimilarity"):
        super(FrameSimilarity, self).__init__(name=name)

        self.projection = tf.keras.layers.Dense(similarity_dim, use_bias=use_bias, activation=None)
        self.fc = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu)

        self.lookup_window = lookup_window
        self.stop_gradient = stop_gradient
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def call(self, inputs):
        x = tf.concat([
            tf.math.reduce_mean(x, axis=[2, 3]) for x in inputs
        ], axis=2)

        if self.stop_gradient:
            x = tf.stop_gradient(x)

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


@gin.configurable(whitelist=["filters", "delta_scale", "loss_weight"])
class ConvexCombinationRegularization(tf.keras.layers.Layer):

    def __init__(self, filters=32, delta_scale=10., loss_weight=0.01, name="ConvexCombinationRegularization"):
        super(ConvexCombinationRegularization, self).__init__(name=name)

        self.projection = tf.keras.layers.Conv3D(filters, kernel_size=1, dilation_rate=1, padding="SAME",
                                                 activation=tf.nn.relu, use_bias=True)
        self.features = tf.keras.layers.Conv3D(filters * 2, kernel_size=(3, 3, 3), dilation_rate=1, padding="SAME",
                                               activation=tf.nn.relu, use_bias=True)
        self.dense = tf.keras.layers.Dense(1, activation=None, use_bias=True)
        self.loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        self.delta_scale = delta_scale
        self.loss_weight = loss_weight

    def call(self, image_inputs, feature_inputs):
        x = feature_inputs
        x = self.projection(x)

        batch_size = tf.shape(x)[0]
        window_size = tf.shape(x)[1]

        first_frame = tf.tile(x[:, :1], [1, window_size, 1, 1, 1])
        last_frame = tf.tile(x[:, -1:], [1, window_size, 1, 1, 1])

        x = tf.concat([x, first_frame, last_frame], -1)
        x = self.features(x)

        x = tf.math.reduce_mean(x, axis=[2, 3])
        alpha = self.dense(x)

        first_img = tf.tile(image_inputs[:, :1], [1, window_size, 1, 1, 1])
        last_img = tf.tile(image_inputs[:, -1:], [1, window_size, 1, 1, 1])

        alpha_ = tf.nn.sigmoid(alpha)
        alpha_ = tf.reshape(alpha_, [batch_size, window_size, 1, 1, 1])
        predictions_ = (alpha_ * first_img + (1 - alpha_) * last_img)

        loss_ = self.loss(y_true=image_inputs / self.delta_scale, y_pred=predictions_ / self.delta_scale)
        loss_ = self.loss_weight * tf.math.reduce_mean(loss_)
        return alpha, loss_


@gin.configurable(whitelist=["lookup_window", "output_dim"])
class ColorHistograms(tf.keras.layers.Layer):

    def __init__(self, lookup_window=101, output_dim=None, name="ColorHistograms"):
        super(ColorHistograms, self).__init__(name=name)

        self.fc = tf.keras.layers.Dense(output_dim, activation=tf.nn.relu) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        frames = tf.cast(frames, tf.int32)

        def get_bin(frames):
            # returns 0 .. 511
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = tf.bitwise.right_shift(R, 5), tf.bitwise.right_shift(G, 5), tf.bitwise.right_shift(B, 5)
            return tf.bitwise.left_shift(R, 6) + tf.bitwise.left_shift(G, 3) + B

        batch_size, time_window, height, width = tf.shape(frames)[0], tf.shape(frames)[1], tf.shape(frames)[2], \
                                                 tf.shape(frames)[3]
        no_channels = frames.shape[-1]
        assert no_channels == 3 or no_channels == 6
        if no_channels == 3:
            frames_flatten = tf.reshape(frames, [batch_size * time_window, height * width, 3])
        else:
            frames_flatten = tf.reshape(frames, [batch_size * time_window, height * width * 2, 3])

        binned_values = get_bin(frames_flatten)
        frame_bin_prefix = tf.bitwise.left_shift(tf.range(batch_size * time_window), 9)[:, tf.newaxis]
        binned_values = binned_values + frame_bin_prefix

        ones = tf.ones_like(binned_values, dtype=tf.int32)
        histograms = tf.math.unsorted_segment_sum(ones, binned_values, batch_size * time_window * 512)
        histograms = tf.reshape(histograms, [batch_size, time_window, 512])

        histograms_normalized = tf.cast(histograms, tf.float32)
        histograms_normalized = histograms_normalized / tf.linalg.norm(histograms_normalized, axis=2, keepdims=True)
        return histograms_normalized

    def call(self, inputs):
        x = self.compute_color_histograms(inputs)

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

        if self.fc is not None:
            return self.fc(similarities)
        return similarities
