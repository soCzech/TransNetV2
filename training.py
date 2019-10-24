import os
import gin
import glob
import argparse
import datetime
from PIL import Image
import tensorflow as tf
import gin.tf.external_configurables

import models
import transnet
import metrics_utils
import input_processing
import visualization_utils


@gin.configurable("options")
def get_options_dict(n_epochs=None,
                     log_dir=gin.REQUIRED,
                     log_name=gin.REQUIRED,
                     trn_files=gin.REQUIRED,
                     tst_files=gin.REQUIRED,
                     input_shape=gin.REQUIRED,
                     test_only=False,
                     restore=None,
                     restore_resnet_features=None,
                     original_transnet=False):
    trn_files_ = []
    for fn in trn_files:
        trn_files_.extend(glob.glob(fn))

    tst_files_ = {}
    for k, v in tst_files.items():
        tst_files_[k] = []
        for fn in v:
            tst_files_[k].extend(glob.glob(fn))

    log_dir = os.path.join(log_dir, log_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir)

    config_str = gin.config_str().replace("# ", "### ").split("\n")
    config_str = "\n\n".join([l for l in config_str if not l.startswith("### =====")])

    with summary_writer.as_default():
        tf.summary.text("config", config_str, step=0)
    with open(os.path.join(log_dir, "config.gin"), "w") as f:
        f.write(config_str)

    return {
        "n_epochs": n_epochs,
        "log_dir": log_dir,
        "summary_writer": summary_writer,
        "trn_files": trn_files_,
        "tst_files": tst_files_,
        "input_shape": input_shape,
        "test_only": test_only,
        "restore": restore,
        "restore_resnet_features": restore_resnet_features,
        "original_transnet": original_transnet
    }


@gin.configurable("training", blacklist=["net", "summary_writer"])
class Trainer:

    def __init__(self, net, summary_writer,
                 optimizer=None,
                 log_freq=None,
                 grad_clipping=10.):
        self.net = net
        self.summary_writer = summary_writer
        self.optimizer = optimizer() if optimizer is not None else None
        self.log_freq = log_freq
        self.grad_clipping = grad_clipping
        self.mean_metrics = dict([(name, tf.keras.metrics.Mean(name=name, dtype=tf.float32)) for name in
                                  ["loss/total", "loss/one_hot_loss", "loss/many_hot_loss", "loss/l2_loss"]])

    @gin.configurable("loss", blacklist=["one_hot_pred", "one_hot_gt", "many_hot_pred", "many_hot_gt"])
    def compute_loss(self, one_hot_pred, one_hot_gt, many_hot_pred=None, many_hot_gt=None,
                     transition_weight=1.,
                     many_hot_loss_weight=0.,
                     l2_loss_weight=0.):

        with tf.name_scope("losses"):
            one_hot_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=one_hot_pred[:, :, 0],
                                                                   labels=tf.cast(one_hot_gt, tf.float32))
            if transition_weight != 1:
                one_hot_loss *= 1 + tf.cast(one_hot_gt, tf.float32) * (transition_weight - 1)
            one_hot_loss = tf.reduce_mean(one_hot_loss)

            many_hot_loss = 0.
            if many_hot_loss_weight != 0. and many_hot_pred is not None:
                many_hot_loss = many_hot_loss_weight * tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=many_hot_pred[:, :, 0],
                                                            labels=tf.cast(many_hot_gt, tf.float32)))

            l2_loss = 0.
            if l2_loss_weight != 0.:
                l2_loss = l2_loss_weight * tf.add_n([tf.nn.l2_loss(v) for v in self.net.trainable_weights],
                                                    name="l2_loss")
            total_loss = one_hot_loss + many_hot_loss + l2_loss

            return total_loss, {"loss/total": total_loss,
                                "loss/one_hot_loss": one_hot_loss,
                                "loss/many_hot_loss": many_hot_loss,
                                "loss/l2_loss": l2_loss}

    @tf.function(autograph=False)
    def train_batch(self, frame_sequence, one_hot_gt, many_hot_gt, run_summaries=False):
        with tf.GradientTape() as tape:
            one_hot_pred = self.net(frame_sequence)
            many_hot_pred = None
            if isinstance(one_hot_pred, tuple):
                one_hot_pred, many_hot_pred = one_hot_pred

            total_loss, losses_dict = self.compute_loss(one_hot_pred, one_hot_gt,
                                                        many_hot_pred, many_hot_gt)

        grads = tape.gradient(total_loss, self.net.trainable_weights)
        with tf.name_scope("grad_check"):
            grads = [
                tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
                for g in grads]
            grads, grad_norm = tf.clip_by_global_norm(grads, self.grad_clipping)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))

        for loss_name, loss_value in losses_dict.items():
            self.mean_metrics[loss_name].update_state(loss_value)

        with self.summary_writer.as_default():
            tf.summary.scalar("grads/norm", grad_norm, step=self.optimizer.iterations)

        if not run_summaries:
            return

        with self.summary_writer.as_default():
            for grad, var in zip(grads, self.net.trainable_weights):
                tf.summary.histogram("grad/" + var.name, grad, step=self.optimizer.iterations)
                tf.summary.histogram("var/" + var.name, var.value(), step=self.optimizer.iterations)

            for loss_name, loss_value in losses_dict.items():
                tf.summary.scalar(loss_name, self.mean_metrics[loss_name].result(), step=self.optimizer.iterations)
                self.mean_metrics[loss_name].reset_states()

        return one_hot_pred, many_hot_pred, self.optimizer.iterations

    def train_epoch(self, dataset, logit_fc=tf.sigmoid):
        print("\nTraining")
        for metric in self.mean_metrics.values():
            metric.reset_states()

        for i, (frame_sequence, one_hot_gt, many_hot_gt) in dataset.enumerate():
            if i % self.log_freq == 0:
                one_hot_pred, many_hot_pred, step = self.train_batch(
                    frame_sequence, one_hot_gt, many_hot_gt, run_summaries=True)

                with self.summary_writer.as_default():
                    visualizations = visualization_utils.visualize_predictions(
                        frame_sequence.numpy(), logit_fc(one_hot_pred).numpy(), one_hot_gt.numpy(),
                        logit_fc(many_hot_pred).numpy() if many_hot_pred is not None else None, many_hot_gt.numpy())
                    tf.summary.image("train/visualization", visualizations, step=step)
            else:
                self.train_batch(frame_sequence, one_hot_gt, many_hot_gt, run_summaries=False)
            print("\r", i.numpy(), end="")

    @tf.function(autograph=False)
    def test_batch(self, frame_sequence, one_hot_gt, many_hot_gt):
        one_hot_pred = self.net(frame_sequence)
        many_hot_pred = None
        if isinstance(one_hot_pred, tuple):
            one_hot_pred, many_hot_pred = one_hot_pred

        total_loss, losses_dict = self.compute_loss(one_hot_pred, one_hot_gt,
                                                    many_hot_pred, many_hot_gt)

        for loss_name, loss_value in losses_dict.items():
            self.mean_metrics[loss_name].update_state(loss_value)

        return one_hot_pred, many_hot_pred

    def test_epoch(self, datasets, epoch_no, save_visualization_to=None, trace=False, logit_fc=tf.sigmoid):
        for metric in self.mean_metrics.values():
            metric.reset_states()

        for ds_name, dataset in datasets:
            print("\nEvaluating", ds_name.upper())
            one_hot_gt_list, one_hot_pred_list = [], []

            for i, (frame_sequence, one_hot_gt, many_hot_gt) in dataset.enumerate():
                if trace:
                    tf.summary.trace_on(graph=True, profiler=False)
                one_hot_pred, many_hot_pred = self.test_batch(frame_sequence, one_hot_gt, many_hot_gt)
                with self.summary_writer.as_default():
                    if trace:
                        tf.summary.trace_export(name="graph", step=0)
                        trace = False

                one_hot_gt_list.extend(one_hot_gt.numpy().flatten().tolist())
                one_hot_pred_list.extend(logit_fc(one_hot_pred).numpy().flatten().tolist())

                print("\r", i.numpy(), end="")
                if i != 0 or save_visualization_to is None:
                    continue

                with self.summary_writer.as_default():
                    visualizations = visualization_utils.visualize_predictions(
                        frame_sequence.numpy(), logit_fc(one_hot_pred).numpy(), one_hot_gt.numpy(),
                        logit_fc(many_hot_pred).numpy() if many_hot_pred is not None else None, many_hot_gt.numpy())
                    tf.summary.image("test/{}/visualization".format(ds_name), visualizations, step=epoch_no)

                    for idx, img in enumerate(visualizations):
                        Image.fromarray(img).save("{}_{}_{:02d}.png".format(save_visualization_to, ds_name, idx))

            with self.summary_writer.as_default():
                for loss_name, loss in self.mean_metrics.items():
                    tf.summary.scalar("test/{}/{}".format(ds_name, loss_name), loss.result(), step=epoch_no)

                metrics_utils.create_scene_based_summaries(one_hot_pred_list, one_hot_gt_list,
                                                           prefix="test/" + ds_name, step=epoch_no)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransNet")
    parser.add_argument("config", help="path to config")
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    options = get_options_dict()

    trn_ds = input_processing.train_pipeline(options["trn_files"]) if len(options["trn_files"]) > 0 else None
    tst_ds = [(name, input_processing.test_pipeline(files))
              for name, files in options["tst_files"].items()]

    if options["original_transnet"]:
        net = models.OriginalTransNet()
        logit_fc = lambda x: tf.nn.softmax(x)[:, :, 1]
    else:
        net = transnet.TransNetV2()
        logit_fc = tf.sigmoid

    net(tf.zeros([1] + options["input_shape"], tf.float32))
    trainer = Trainer(net, options["summary_writer"])

    if options["restore_resnet_features"] is not None:
        net.resnet_layers.restore_me(options["restore_resnet_features"])
        print("ResNet weights restored from", options["restore_resnet_features"])

    if options["restore"] is not None:
        net.load_weights(options["restore"])
        print("Weights restored from", options["restore"])

    if options["test_only"]:
        trainer.test_epoch(tst_ds, 0, os.path.join(options["log_dir"], "visualization-00"), trace=True,
                           logit_fc=logit_fc)
        exit()

    for epoch in range(1, options["n_epochs"] + 1):
        trainer.train_epoch(trn_ds, logit_fc=logit_fc)
        net.save_weights(os.path.join(options["log_dir"], "weights-{}.h5".format(epoch)))

        trainer.test_epoch(tst_ds, epoch, os.path.join(options["log_dir"], "visualization-{:02d}".format(epoch)),
                           trace=epoch == 1, logit_fc=logit_fc)
