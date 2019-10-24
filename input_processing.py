import gin
import tensorflow as tf


@gin.configurable(blacklist=["filenames"])
def train_pipeline(filenames,
                   shuffle_buffer=100,
                   shot_len=100,
                   frame_width=48,
                   frame_height=27,
                   batch_size=16,
                   repeat=False):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(parse_train_sample,
                                                                                         num_parallel_calls=1),
                       cycle_length=8,
                       block_length=16,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.padded_batch(2, ([shot_len, frame_height, frame_width, 3], []), drop_remainder=True)
    ds = ds.map(concat_shots, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def filter_(*args):
        return args[-1]

    def map_(*args):
        return args[:-1]

    ds = ds.filter(filter_).map(map_)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(2)
    return ds


@tf.function
@gin.configurable(blacklist=["sample"])
def parse_train_sample(sample,
                       shot_len=None,
                       frame_width=48,
                       frame_height=27):
    features = tf.io.parse_single_example(sample, features={
        "scene": tf.io.FixedLenFeature([], tf.string),
        "length": tf.io.FixedLenFeature([], tf.int64)
    })
    length = tf.cast(features["length"], tf.int32)

    scene = tf.io.decode_raw(features["scene"], tf.uint8)
    scene = tf.reshape(scene, [length, frame_height, frame_width, 3])

    shot_start = tf.random.uniform([], minval=0, maxval=tf.maximum(1, length - shot_len), dtype=tf.int32)
    shot_end = shot_start + shot_len
    scene = scene[shot_start:shot_end]

    scene = tf.cast(scene, dtype=tf.float32)
    scene = augment_shot(scene)
    return scene, tf.shape(scene)[0]  # [<SHOT_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3]


@tf.function
@gin.configurable(blacklist=["shot"])
def augment_shot(shot,
                 up_down_flip_prob=.2,
                 left_right_flip_prob=.5,
                 adjust_saturation=True,
                 adjust_contrast=True,
                 adjust_brightness=True,
                 adjust_hue=True):

    shot = tf.cond(tf.random.uniform([]) < up_down_flip_prob,
                   lambda: tf.image.flip_up_down(shot), lambda: shot)

    shot = tf.cond(tf.random.uniform([]) < left_right_flip_prob,
                   lambda: tf.image.flip_left_right(shot), lambda: shot)

    if adjust_saturation:
        shot = tf.image.adjust_saturation(shot, saturation_factor=tf.random.uniform([], minval=0.8, maxval=1.2))
    if adjust_contrast:
        shot = tf.image.adjust_contrast(shot, contrast_factor=tf.random.uniform([], minval=0.8, maxval=1.2))
    if adjust_brightness:
        shot = tf.image.adjust_brightness(shot, delta=tf.random.uniform([], minval=-0.1, maxval=0.1))
    if adjust_hue:
        shot = tf.image.adjust_hue(shot, delta=tf.random.uniform([], minval=-0.1, maxval=0.1))
    return shot


@tf.function
@gin.configurable(blacklist=["shots", "lens"])
def concat_shots(shots,
                 lens,
                 shot_len=None,
                 color_transfer_prob=0.3,
                 transition_min_len=2,
                 transition_max_len=60,
                 hard_cut_prob=0.4,
                 cutout_prob=0.3):
    assert transition_min_len % 2 == 0 and transition_min_len >= 2, "`transition_min_len` must be even"
    assert transition_max_len % 2 == 0, "`transition_max_len` must be even"
    shot1 = shots[0][:lens[0]]
    shot2 = shots[1][:lens[1]]

    if color_transfer_prob > 0:
        shot2 = tf.cond(tf.random.uniform([]) < color_transfer_prob,
                        lambda: color_transfer(source=shot1, target=shot2), lambda: shot2)

    transition_boundary = tf.random.uniform([], maxval=shot_len, dtype=tf.int32)  # {0, ..., shot_len - 1}
    # convert transition boundary to vector with 1 at the boundary and 0 otherwise
    one_hot_gt = tf.one_hot(transition_boundary, shot_len, dtype=tf.int32)  # [SHOT_LENGTH]

    # hard_cut
    hard_cut = tf.cast(tf.range(shot_len) <= transition_boundary, dtype=tf.float32)

    # dissolve
    dis_len = tf.random.uniform([], minval=transition_min_len // 2,
                                maxval=(transition_max_len // 2) + 1, dtype=tf.int32)
    dis_kernel = tf.linspace(1., 0., dis_len * 2 + 2)[1:-1]
    dis_left_win = tf.minimum(dis_len - 1, transition_boundary)
    dis_right_win = tf.minimum(dis_len, (shot_len - 1) - transition_boundary)
    dissolve = tf.concat([
        tf.ones([transition_boundary - dis_left_win], dtype=tf.float32),
        dis_kernel[dis_len - dis_left_win - 1:dis_len + dis_right_win],
        tf.zeros([shot_len - (transition_boundary + dis_right_win + 1)], dtype=tf.float32)
    ], axis=0)
    dissolve_trans = tf.reshape(tf.cast(
        tf.logical_and(tf.not_equal(dissolve, 0.), tf.not_equal(dissolve, 1.)), tf.int32
    ), [shot_len])

    # switch between hard cut and dissolve
    transition, many_hot_gt = tf.cond(tf.random.uniform([]) < hard_cut_prob,
                                      lambda: (hard_cut, one_hot_gt),
                                      lambda: (dissolve, dissolve_trans))

    # add together two shots
    transition = tf.reshape(transition, [shot_len, 1, 1, 1])

    many_hot_gt_indices = tf.cast(tf.where(many_hot_gt), tf.int32)
    shot1_min_len = tf.reduce_max(many_hot_gt_indices)
    shot2_min_len = shot_len - tf.reduce_min(many_hot_gt_indices)

    shot1_pad_start = tf.maximum(shot1_min_len - lens[0], 0)
    shot1_pad_end = tf.maximum(shot_len - (lens[0] + shot1_pad_start), 0)
    shot1 = tf.pad(shot1, [[shot1_pad_start, shot1_pad_end], [0, 0], [0, 0], [0, 0]])

    shot2_pad_end = tf.maximum(shot2_min_len - lens[1], 0)
    shot2_pad_start = tf.maximum(shot_len - (lens[1] + shot2_pad_end), 0)
    shot2 = tf.pad(shot2, [[shot2_pad_start, shot2_pad_end], [0, 0], [0, 0], [0, 0]])

    shot = shot1 * transition + shot2 * (1 - transition)  # [SHOT_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3]

    shot = tf.cond(tf.random.uniform([]) < cutout_prob,
                   lambda: cutout(shot), lambda: shot)
    return shot, one_hot_gt, many_hot_gt, tf.maximum(shot1_pad_start, shot2_pad_end) == 0


@tf.function
@gin.configurable(blacklist=["shot"])
def cutout(shot,
           min_width_fraction=1/4,
           min_height_fraction=1/4,
           max_width_fraction=2/3,
           max_height_fraction=2/3):
    frame_height, frame_width = tf.shape(shot)[1], tf.shape(shot)[2]
    frame_height_float, frame_width_float = tf.cast(frame_height, tf.float32), tf.cast(frame_width, tf.float32)

    height = tf.random.uniform([],
                               tf.cast(frame_height_float * min_height_fraction, tf.int32),
                               tf.cast(frame_height_float * max_height_fraction, tf.int32),
                               tf.int32)
    width = tf.random.uniform([],
                              tf.cast(frame_width_float * min_width_fraction, tf.int32),
                              tf.cast(frame_width_float * max_width_fraction, tf.int32),
                              tf.int32)

    left = tf.random.uniform([], 0, frame_width - width, tf.int32)
    top = tf.random.uniform([], 0, frame_height - height, tf.int32)

    bottom = tf.minimum(top + height, frame_height)
    right = tf.minimum(left + width, frame_width)

    t = tf.random.uniform([1, height, width, 3], 0, 255., dtype=tf.float32)

    random_patch = tf.pad(t, [[0, 0], [top, frame_height - bottom], [left, frame_width - right], [0, 0]])
    mask = tf.pad(tf.zeros([1, height, width, 1]),
                  [[0, 0], [top, frame_height - bottom], [left, frame_width - right], [0, 0]], constant_values=1.)
    return random_patch + shot * mask


@tf.function
def color_transfer(source, target, name="color_transfer"):
    # based on https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
    # transfers based from https://github.com/xahidbuffon/rgb-lab-conv
    with tf.name_scope(name):
        source = rgb_to_lab(source)
        target = rgb_to_lab(target)

        src_mean, src_var = tf.nn.moments(source, axes=(0, 1, 2), keepdims=True)
        src_std = tf.sqrt(src_var)
        tar_mean, tar_var = tf.nn.moments(target, axes=(0, 1, 2), keepdims=True)
        tar_std = tf.sqrt(tar_var)

        # ensure reasonable scaling and prevent division by zero
        src_std = tf.maximum(src_std, 1)
        tar_std = tf.maximum(tar_std, 1)

        target_shifted = (target - tar_mean) * (src_std / tar_std) + src_mean

        lab_min = tf.constant([[[[0, -86.185, -107.863]]]])
        lab_max = tf.constant([[[[100, 98.254, 94.482]]]])
        target_clipped = tf.clip_by_value(target_shifted, lab_min, lab_max)

        return lab_to_rgb(target_clipped)


def rgb_to_lab(rgb):
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(rgb, [-1, 3]) / 255.
        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = 1 - linear_mask
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, tf.constant([[1/0.950456, 1.0, 1/1.088754]], tf.float32))
            # fix when values -0.0001 result in Nan if raised to 1/3
            xyz_normalized_pixels = tf.maximum(xyz_normalized_pixels, 0)

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = 1 - linear_mask
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0],  # fx
                [116.0, -500.0,  200.0],  # fy
                [  0.0,    0.0, -200.0],  # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(rgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])
        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0],  # l
                [1/500.0,     0.0,      0.0],  # a
                [    0.0,     0.0, -1/200.0],  # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = 1 - linear_mask
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434],  # x
                [-1.5371385,  1.8760108, -0.2040259],  # y
                [-0.4985314,  0.0415560,  1.0572252],  # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = 1 - linear_mask
            rgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(rgb_pixels, tf.shape(lab)) * 255.


@gin.configurable(blacklist=["filenames"])
def test_pipeline(filenames,
                  shot_len=100,
                  batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(
            x, compression_type="GZIP").map(parse_test_sample,
                                            num_parallel_calls=1).batch(shot_len, drop_remainder=True),
        cycle_length=8,
        block_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(2)
    return ds


@tf.function
@gin.configurable(blacklist=["sample"])
def parse_test_sample(sample,
                      frame_width=48,
                      frame_height=27):
    features = tf.io.parse_single_example(sample, features={
        "frame": tf.io.FixedLenFeature([], tf.string),
        "is_one_hot_transition": tf.io.FixedLenFeature([], tf.int64),
        "is_many_hot_transition": tf.io.FixedLenFeature([], tf.int64)
    })

    frame = tf.io.decode_raw(features["frame"], tf.uint8)
    frame = tf.reshape(frame, [frame_height, frame_width, 3])

    one_hot = tf.cast(features["is_one_hot_transition"], tf.int32)
    many_hot = tf.cast(features["is_many_hot_transition"], tf.int32)

    frame = tf.cast(frame, dtype=tf.float32)
    return frame, one_hot, many_hot
