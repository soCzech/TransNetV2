import gin
import tensorflow as tf


@gin.configurable(blacklist=["filenames"])
def train_pipeline(filenames,
                   shuffle_buffer=100,
                   shot_len=100,
                   frame_width=48,
                   frame_height=27,
                   batch_size=16,
                   repeat=False,
                   no_channels=3):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP").map(parse_train_sample,
                                                                                         num_parallel_calls=1),
                       cycle_length=8,
                       block_length=16,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.padded_batch(2, ([shot_len, frame_height, frame_width, no_channels], []), drop_remainder=True)
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


@gin.configurable(blacklist=["filenames"])
def train_transition_pipeline(filenames,
                              shuffle_buffer=100,
                              batch_size=16,
                              repeat=False):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.shuffle(len(filenames))
    ds = ds.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
                       cycle_length=8,
                       block_length=16,
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(parse_train_transition_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(2)
    return ds


@tf.function
@gin.configurable(blacklist=["sample"])
def parse_train_transition_sample(sample,
                                  shot_len=None,
                                  frame_width=48,
                                  frame_height=27):
    features = tf.io.parse_single_example(sample, features={
        "scene": tf.io.FixedLenFeature([], tf.string),
        "one_hot": tf.io.FixedLenFeature([], tf.string),
        "many_hot": tf.io.FixedLenFeature([], tf.string),
        "length": tf.io.FixedLenFeature([], tf.int64)
    })
    length = tf.cast(features["length"], tf.int32)

    scene = tf.io.decode_raw(features["scene"], tf.uint8)
    scene = tf.reshape(scene, [length, frame_height, frame_width, 3])

    one_hot = tf.io.decode_raw(features["one_hot"], tf.uint8)
    many_hot = tf.io.decode_raw(features["many_hot"], tf.uint8)

    shot_start = tf.random.uniform([], minval=0, maxval=length - shot_len, dtype=tf.int32)
    shot_end = shot_start + shot_len

    scene = tf.reshape(scene[shot_start:shot_end], [shot_len, frame_height, frame_width, 3])
    scene = tf.cast(scene, dtype=tf.float32)
    scene = augment_shot(scene)

    one_hot = tf.cast(tf.reshape(one_hot[shot_start:shot_end], [shot_len]), tf.int32)
    many_hot = tf.cast(tf.reshape(many_hot[shot_start:shot_end], [shot_len]), tf.int32)

    return scene, one_hot, many_hot


@tf.function
@gin.configurable(blacklist=["sample"])
def parse_train_sample(sample,
                       shot_len=None,
                       frame_width=48,
                       frame_height=27,
                       sudden_color_change_prob=0.,
                       spacial_augmentation=False,
                       original_width=None,
                       original_height=None,
                       no_channels=3):
    assert no_channels == 3 or no_channels == 6

    features = tf.io.parse_single_example(sample, features={
        "scene": tf.io.FixedLenFeature([], tf.string),
        "length": tf.io.FixedLenFeature([], tf.int64)
    })
    length = tf.cast(features["length"], tf.int32)

    original_width = original_width if spacial_augmentation else frame_width
    original_height = original_height if spacial_augmentation else frame_height

    scene = tf.io.decode_raw(features["scene"], tf.uint8)
    scene = tf.reshape(scene, [length, original_height, original_width, no_channels])

    shot_start = tf.random.uniform([], minval=0, maxval=tf.maximum(1, length - shot_len), dtype=tf.int32)
    shot_end = shot_start + shot_len
    scene = scene[shot_start:shot_end]

    scene = tf.cast(scene, dtype=tf.float32)

    if sudden_color_change_prob != 0.:
        assert no_channels == 3  # not implemented

        def color_change(shot_):
            bound = tf.random.uniform([], minval=1, maxval=tf.shape(shot_)[0], dtype=tf.int32)
            start, end = shot_[:bound], shot_[bound:]
            start = augment_shot(start, up_down_flip_prob=0., left_right_flip_prob=0.)
            return tf.concat([start, end], axis=0)

        scene = tf.cond(tf.random.uniform([]) < sudden_color_change_prob,
                        lambda: color_change(scene), lambda: scene)

    if spacial_augmentation:
        assert no_channels == 3  # not implemented
        scene = augment_shot_spacial(scene, frame_width, frame_height)

    scene = augment_shot(scene, no_channels=no_channels)
    return scene, tf.shape(scene)[0]  # [<SHOT_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3]


@tf.function
@gin.configurable(blacklist=["shot", "target_width", "target_height"])
def augment_shot_spacial(shot, target_width, target_height,
                         random_shake_prob=0.3,
                         random_shake_max_size=15,
                         clip_left_right=20,
                         clip_top_bottom=10):
    # random shake
    def random_shake(shot_):
        bound = tf.random.uniform([], minval=1, maxval=tf.shape(shot_)[0], dtype=tf.int32)
        shake = tf.random.uniform([], minval=1, maxval=random_shake_max_size, dtype=tf.int32)

        start, end = shot[:bound], shot[bound:]
        start, end = tf.cond(tf.random.uniform([]) < 0.5,
                             lambda: (start[:, shake:], end[:, :-shake]),
                             lambda: (start[:, :-shake], end[:, shake:]))
        return tf.concat([start, end], axis=0)

    shot = tf.cond(tf.random.uniform([]) < random_shake_prob,
                   lambda: random_shake(shot), lambda: shot)

    if clip_left_right != 0 or clip_top_bottom != 0:
        left = tf.random.uniform([], minval=0, maxval=clip_left_right, dtype=tf.int32)
        right = tf.random.uniform([], minval=0, maxval=clip_left_right, dtype=tf.int32)

        top = tf.random.uniform([], minval=0, maxval=clip_top_bottom, dtype=tf.int32)
        bottom = tf.random.uniform([], minval=0, maxval=clip_top_bottom, dtype=tf.int32)

        shot = shot[:, top:tf.shape(shot)[1] - bottom, left:tf.shape(shot)[2] - right]

    shot = tf.image.resize(shot, [target_height, target_width])
    return shot


@tf.function
@gin.configurable(blacklist=["shot", "no_channels"])
def augment_shot(shot,
                 up_down_flip_prob=.2,
                 left_right_flip_prob=.5,
                 adjust_saturation=True,
                 adjust_contrast=True,
                 adjust_brightness=True,
                 adjust_hue=True,
                 equalize_prob=0.,
                 posterize_prob=0.,
                 posterize_min_bits=2,
                 color_prob=0.,
                 color_min_val=0.3,
                 color_max_val=1.7,
                 no_channels=3):

    if no_channels != 3:
        shot_len, shot_height, shot_width = tf.shape(shot)[0], tf.shape(shot)[1], tf.shape(shot)[2]
        shot = tf.reshape(shot, [shot_len, shot_height, shot_width * 2, 3])

    shot = tf.cond(tf.random.uniform([]) < up_down_flip_prob,
                   lambda: tf.image.flip_up_down(shot), lambda: shot)

    assert no_channels == 3 or left_right_flip_prob == 0.
    shot = tf.cond(tf.random.uniform([]) < left_right_flip_prob,
                   lambda: tf.image.flip_left_right(shot), lambda: shot)

    shot = shot / 255.
    if adjust_saturation:
        shot = tf.image.adjust_saturation(shot, saturation_factor=tf.random.uniform([], minval=0.8, maxval=1.2))
    if adjust_contrast:
        shot = tf.image.adjust_contrast(shot, contrast_factor=tf.random.uniform([], minval=0.8, maxval=1.2))
        shot = tf.clip_by_value(shot, 0., 1.)
    if adjust_brightness:
        shot = tf.image.adjust_brightness(shot, delta=tf.random.uniform([], minval=-0.1, maxval=0.1))
    if adjust_hue:
        shot = tf.image.adjust_hue(shot, delta=tf.random.uniform([], minval=-0.1, maxval=0.1))

    shot = tf.clip_by_value(shot, 0., 1.) * 255.

    if color_prob != 0.:
        factor = tf.random.uniform([], minval=color_min_val, maxval=color_max_val)
        shot = tf.cond(tf.random.uniform([]) < color_prob,
                       lambda: pil_color(shot, factor), lambda: shot)

    if equalize_prob != 0. or posterize_prob != 0.:
        shot = tf.cast(shot, tf.uint8)
        shot = tf.cond(tf.random.uniform([]) < equalize_prob,
                       lambda: pil_equalize(shot), lambda: shot)

        bits = tf.random.uniform([], minval=posterize_min_bits, maxval=7, dtype=tf.int32)
        shot = tf.cond(tf.random.uniform([]) < posterize_prob,
                       lambda: pil_posterize(shot, bits=tf.cast(bits, tf.uint8)), lambda: shot)
        shot = tf.cast(shot, tf.float32)

    if no_channels != 3:
        shot = tf.reshape(shot, [shot_len, shot_height, shot_width, 6])
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
                 cutout_prob=0.3,
                 advanced_shot_trans_prob=0.,
                 no_channels=3):
    assert transition_min_len % 2 == 0 and transition_min_len >= 2, "`transition_min_len` must be even"
    assert transition_max_len % 2 == 0, "`transition_max_len` must be even"
    shot1 = shots[0][:lens[0]]
    shot2 = shots[1][:lens[1]]

    if color_transfer_prob > 0:
        assert no_channels == 3  # not implemented
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
    is_dissolve = tf.random.uniform([]) > hard_cut_prob
    transition, many_hot_gt = tf.cond(is_dissolve,
                                      lambda: (dissolve, dissolve_trans),
                                      lambda: (hard_cut, one_hot_gt))

    # pad shots to full length if they are smaller
    many_hot_gt_indices = tf.cast(tf.where(many_hot_gt), tf.int32)
    shot1_min_len = tf.reduce_max(many_hot_gt_indices)
    shot2_min_len = shot_len - tf.reduce_min(many_hot_gt_indices)

    shot1_pad_start = tf.maximum(shot1_min_len - lens[0], 0)
    shot1_pad_end = tf.maximum(shot_len - (lens[0] + shot1_pad_start), 0)
    shot1 = tf.pad(shot1, [[shot1_pad_start, shot1_pad_end], [0, 0], [0, 0], [0, 0]])

    shot2_pad_end = tf.maximum(shot2_min_len - lens[1], 0)
    shot2_pad_start = tf.maximum(shot_len - (lens[1] + shot2_pad_end), 0)
    shot2 = tf.pad(shot2, [[shot2_pad_start, shot2_pad_end], [0, 0], [0, 0], [0, 0]])

    def basic_shot_transitions(shot1, shot2, trans_interpolation):
        # add together two shots
        trans_interpolation = tf.reshape(trans_interpolation, [tf.shape(shot1)[0], 1, 1, 1])
        return shot1 * trans_interpolation + shot2 * (1 - trans_interpolation)

    # [SHOT_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
    shot = tf.cond(tf.logical_and(is_dissolve, tf.random.uniform([]) < advanced_shot_trans_prob),
                   lambda: advanced_shot_transitions(shot1, shot2, transition),
                   lambda: basic_shot_transitions(shot1, shot2, transition))

    if cutout_prob > 0.:
        assert no_channels == 3  # not implemented
        shot = tf.cond(tf.random.uniform([]) < cutout_prob,
                       lambda: cutout(shot), lambda: shot)
    return shot, one_hot_gt, many_hot_gt, tf.maximum(shot1_pad_start, shot2_pad_end) == 0


@tf.function
def advanced_shot_transitions(shot1, shot2, trans_interpolation):
    # transition in horizontal or vertical direction
    flip_wh = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
    shot1, shot2 = tf.cond(flip_wh,
                           lambda: (tf.transpose(shot1, [0, 2, 1, 3]), tf.transpose(shot2, [0, 2, 1, 3])),
                           lambda: (shot1, shot2))

    # transition from top to bottom or from bottom to top
    flip_fromto = tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool)
    trans_interpolation = tf.cond(flip_fromto, lambda: trans_interpolation, lambda: 1 - trans_interpolation)

    shot_len, shot_height, shot_width = tf.shape(shot1)[0], tf.shape(shot1)[1], tf.shape(shot1)[2]
    # compute gather indices
    time_indices = tf.tile(tf.reshape(tf.range(shot_len), [-1, 1]), [1, shot_height])
    initial_rows = tf.tile(tf.reshape(tf.range(shot_height), [1, -1]), [shot_len, 1])
    row_additions = tf.cast(tf.reshape(trans_interpolation, [-1, 1]) * tf.cast(shot_height, tf.float32), tf.int32)
    indices = tf.stack([time_indices, initial_rows + row_additions], -1)

    # makes the shot move
    shot1_out = tf.gather_nd(tf.concat([shot1, tf.zeros_like(shot1)], 1), indices)
    shot2_out = tf.gather_nd(tf.concat([tf.zeros_like(shot2), shot2], 1), indices)
    # makes the shot stationary
    shot1_mask = tf.gather_nd(tf.concat([tf.ones_like(shot1), tf.zeros_like(shot1)], 1), indices)
    shot2_mask = tf.gather_nd(tf.concat([tf.zeros_like(shot2), tf.ones_like(shot2)], 1), indices)

    # select between moving or stationary variant for each shot
    shot1 = tf.cond(tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool),
                    lambda: shot1_out, lambda: shot1 * shot1_mask)
    shot2 = tf.cond(tf.cast(tf.random.uniform([], maxval=2, dtype=tf.int32), tf.bool),
                    lambda: shot2_out, lambda: shot2 * shot2_mask)

    result = shot1 + shot2

    # flip back if needed
    result = tf.cond(flip_wh, lambda: tf.transpose(result, [0, 2, 1, 3]), lambda: result)
    return result


@tf.function
@gin.configurable(blacklist=["shot"])
def cutout(shot,
           min_width_fraction=1/4,
           min_height_fraction=1/4,
           max_width_fraction=2/3,
           max_height_fraction=2/3,
           cutout_color=None):
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

    if cutout_color is not None:
        t = tf.fill([1, height, width, 3], tf.constant(cutout_color, dtype=tf.float32))
        # t = tf.zeros([1, height, width, 3], dtype=tf.float32) + cutout_color
    else:
        t = tf.random.uniform([1, height, width, 3], 0, 255., dtype=tf.float32)

    random_patch = tf.pad(t, [[0, 0], [top, frame_height - bottom], [left, frame_width - right], [0, 0]])
    mask = tf.pad(tf.zeros([1, height, width, 1]),
                  [[0, 0], [top, frame_height - bottom], [left, frame_width - right], [0, 0]], constant_values=1.)
    return random_patch + shot * mask


@tf.function
def pil_equalize(shot):
    # Implements Equalize function from PIL using TF ops.
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    def scale_channel(im, c):
        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2 and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range. This is done in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.
        # Otherwise, build lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0),
                         lambda: im,
                         lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.uint8)

    # Assumes RGB for now. Scales each channel independently and then stacks the result.
    l, h, w, c = tf.shape(shot)[0], tf.shape(shot)[1], tf.shape(shot)[2], tf.shape(shot)[3]

    shot = tf.reshape(shot, [l * h, w, c])
    s1 = scale_channel(shot, 0)
    s2 = scale_channel(shot, 1)
    s3 = scale_channel(shot, 2)
    shot = tf.stack([s1, s2, s3], 2)
    shot = tf.reshape(shot, [l, h, w, c])
    return shot


def pil_posterize(image, bits):
    # Equivalent of PIL Posterize.
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    shift = 8 - bits
    return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def pil_color(shot, factor):
    # Equivalent of PIL Color.
    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(shot))
    difference = shot - degenerate
    scaled = factor * difference
    return tf.clip_by_value(degenerate + scaled, 0., 255.)


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
                      frame_height=27,
                      no_channels=3):
    features = tf.io.parse_single_example(sample, features={
        "frame": tf.io.FixedLenFeature([], tf.string),
        "is_one_hot_transition": tf.io.FixedLenFeature([], tf.int64),
        "is_many_hot_transition": tf.io.FixedLenFeature([], tf.int64)
    })

    frame = tf.io.decode_raw(features["frame"], tf.uint8)
    frame = tf.reshape(frame, [frame_height, frame_width, no_channels])

    one_hot = tf.cast(features["is_one_hot_transition"], tf.int32)
    many_hot = tf.cast(features["is_many_hot_transition"], tf.int32)

    frame = tf.cast(frame, dtype=tf.float32)
    return frame, one_hot, many_hot
