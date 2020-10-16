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
from bi_tempered_loss import bi_tempered_binary_logistic_loss, tempered_sigmoid
import weight_decay_optimizers
gin.config.external_configurable(weight_decay_optimizers.SGDW, 'weight_decay_optimizers.SGDW')


@gin.configurable("options", blacklist=["create_dir_and_summaries"])
def get_options_dict(n_epochs=None,
                     log_dir=gin.REQUIRED,
                     log_name=gin.REQUIRED,
                     trn_files=gin.REQUIRED,
                     tst_files=gin.REQUIRED,
                     input_shape=gin.REQUIRED,
                     test_only=False,
                     restore=None,
                     restore_resnet_features=None,
                     original_transnet=False,
                     transition_only_trn_files=None,
                     create_dir_and_summaries=True,
                     transition_only_data_fraction=0.3,
                     c3d_net=False,
                     bi_tempered_loss=False,
                     bi_tempered_loss_temp2=1.,
                     learning_rate_schedule=None,
                     learning_rate_decay=None):
    trn_files_ = []
    for fn in trn_files:
        trn_files_.extend(glob.glob(fn))

    if transition_only_trn_files is not None:
        transition_trn_files_ = []
        for fn in transition_only_trn_files:
            transition_trn_files_.extend(glob.glob(fn))

    tst_files_ = {}
    for k, v in tst_files.items():
        tst_files_[k] = []
        for fn in v:
            tst_files_[k].extend(glob.glob(fn))

    log_dir = os.path.join(log_dir, log_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    summary_writer = tf.summary.create_file_writer(log_dir) if create_dir_and_summaries else None

    config_str = gin.config_str().replace("# ", "### ").split("\n")
    config_str = "\n\n".join([l for l in config_str if not l.startswith("### =====")])

    if create_dir_and_summaries:
        with summary_writer.as_default():
            tf.summary.text("config", config_str, step=0)
        with open(os.path.join(log_dir, "config.gin"), "w") as f:
            f.write(gin.config_str())

    print("\n{}\n".format(log_name.upper()))

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
        "original_transnet": original_transnet,
        "transition_only_trn_files": transition_trn_files_ if transition_only_trn_files is not None else None,
        "transition_only_data_fraction": transition_only_data_fraction,
        "c3d_net": c3d_net,
        "bi_tempered_loss": bi_tempered_loss,
        "bi_tempered_loss_temp2": bi_tempered_loss_temp2,
        "learning_rate_schedule": learning_rate_schedule,
        "learning_rate_decay": learning_rate_decay
    }


@gin.configurable("training", blacklist=["net", "summary_writer"])
class Trainer:

    def __init__(self, net, summary_writer,
                 optimizer=None,
                 log_freq=None,
                 grad_clipping=10.,
                 n_batches_per_epoch=None,
                 evaluate_on_middle_frames_only=False):
        self.net = net
        self.summary_writer = summary_writer
        self.optimizer = optimizer() if optimizer is not None else None
        self.log_freq = log_freq
        self.grad_clipping = grad_clipping
        self.n_batches_per_epoch = n_batches_per_epoch
        self.mean_metrics = dict([(name, tf.keras.metrics.Mean(name=name, dtype=tf.float32)) for name in
                                  ["loss/total", "loss/one_hot_loss", "loss/many_hot_loss", "loss/l2_loss",
                                   "loss/comb_reg"]])
        self.results = {}
        self.evaluate_on_middle_frames_only = evaluate_on_middle_frames_only

    @gin.configurable("loss", blacklist=["one_hot_pred", "one_hot_gt", "many_hot_pred", "many_hot_gt", "reg_losses"])
    def compute_loss(self, one_hot_pred, one_hot_gt, many_hot_pred=None, many_hot_gt=None,
                     transition_weight=1.,
                     many_hot_loss_weight=0.,
                     l2_loss_weight=0.,
                     dynamic_weight=None,
                     reg_losses=None,
                     bi_tempered_loss=False,
                     bi_tempered_loss_temp1=1.,
                     bi_tempered_loss_temp2=1.):
        assert not (dynamic_weight and transition_weight != 1)

        one_hot_pred = one_hot_pred[:, :, 0]

        with tf.name_scope("losses"):
            if bi_tempered_loss:
                one_hot_loss = bi_tempered_binary_logistic_loss(activations=one_hot_pred,
                                                                labels=tf.cast(one_hot_gt, tf.float32),
                                                                t1=bi_tempered_loss_temp1, t2=bi_tempered_loss_temp2,
                                                                label_smoothing=0.)
            else:
                one_hot_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=one_hot_pred,
                                                                       labels=tf.cast(one_hot_gt, tf.float32))
            if transition_weight != 1:
                one_hot_loss *= 1 + tf.cast(one_hot_gt, tf.float32) * (transition_weight - 1)
            elif dynamic_weight is not None:
                pred_sigmoid = tf.nn.sigmoid(one_hot_pred)
                trans_weight = 4 * (dynamic_weight - 1) * (pred_sigmoid * pred_sigmoid - pred_sigmoid + 0.25)
                trans_weight = tf.where(pred_sigmoid < 0.5, trans_weight, 0)
                trans_weight = tf.stop_gradient(trans_weight)
                one_hot_loss *= 1 + tf.cast(one_hot_gt, tf.float32) * trans_weight

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
            losses = {
                "loss/one_hot_loss": one_hot_loss,
                "loss/many_hot_loss": many_hot_loss,
                "loss/l2_loss": l2_loss
            }

            if reg_losses is not None:
                for name, value in reg_losses.items():
                    if value is not None:
                        total_loss += value
                        losses["loss/" + name] = value
            losses["loss/total"] = total_loss

            return total_loss, losses

    @tf.function(autograph=False)
    def train_batch(self, frame_sequence, one_hot_gt, many_hot_gt, run_summaries=False):
        with tf.GradientTape() as tape:
            one_hot_pred = self.net(frame_sequence, training=True)

            dict_ = {}
            if isinstance(one_hot_pred, tuple):
                one_hot_pred, dict_ = one_hot_pred

            many_hot_pred = dict_.get("many_hot", None)
            alphas = dict_.get("alphas", None)
            comb_reg_loss = dict_.get("comb_reg_loss", None)

            total_loss, losses_dict = self.compute_loss(one_hot_pred, one_hot_gt,
                                                        many_hot_pred, many_hot_gt,
                                                        reg_losses={"comb_reg": comb_reg_loss})

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
            tf.summary.scalar("loss/immediate/total", total_loss, step=self.optimizer.iterations)

        if not run_summaries:
            return

        with self.summary_writer.as_default():
            for grad, var in zip(grads, self.net.trainable_weights):
                tf.summary.histogram("grad/" + var.name, grad, step=self.optimizer.iterations)
                tf.summary.histogram("var/" + var.name, var.value(), step=self.optimizer.iterations)

            for loss_name, loss_value in losses_dict.items():
                tf.summary.scalar(loss_name, self.mean_metrics[loss_name].result(), step=self.optimizer.iterations)
                self.mean_metrics[loss_name].reset_states()
            tf.summary.scalar("learning_rate", self.optimizer.learning_rate, step=self.optimizer.iterations)

        return one_hot_pred, alphas if alphas is not None else many_hot_pred, self.optimizer.iterations

    def train_epoch(self, dataset, logit_fc=tf.sigmoid):
        print("\nTraining")
        for metric in self.mean_metrics.values():
            metric.reset_states()

        for i, (frame_sequence, one_hot_gt, many_hot_gt) in dataset.enumerate():
            if i % self.log_freq == self.log_freq - 1:
                one_hot_pred, many_hot_pred, step = self.train_batch(
                    frame_sequence, one_hot_gt, many_hot_gt, run_summaries=True)

                with self.summary_writer.as_default():
                    visualizations = visualization_utils.visualize_predictions(
                        frame_sequence.numpy()[:, :, :, :, :3], logit_fc(one_hot_pred).numpy(), one_hot_gt.numpy(),
                        logit_fc(many_hot_pred).numpy() if many_hot_pred is not None else None, many_hot_gt.numpy())
                    tf.summary.image("train/visualization", visualizations, step=step)

                for metric in self.mean_metrics.values():
                    metric.reset_states()
            else:
                self.train_batch(frame_sequence, one_hot_gt, many_hot_gt, run_summaries=False)
            print("\r", i.numpy(), end="")
            if self.n_batches_per_epoch is not None and self.n_batches_per_epoch == i:
                break

    @tf.function(autograph=False)
    def test_batch(self, frame_sequence, one_hot_gt, many_hot_gt):
        one_hot_pred = self.net(frame_sequence, training=False)

        dict_ = {}
        if isinstance(one_hot_pred, tuple):
            one_hot_pred, dict_ = one_hot_pred

        many_hot_pred = dict_.get("many_hot", None)
        alphas = dict_.get("alphas", None)
        comb_reg_loss = dict_.get("comb_reg_loss", None)

        total_loss, losses_dict = self.compute_loss(one_hot_pred, one_hot_gt,
                                                    many_hot_pred, many_hot_gt,
                                                    reg_losses={"comb_reg": comb_reg_loss})

        for loss_name, loss_value in losses_dict.items():
            self.mean_metrics[loss_name].update_state(loss_value)

        return one_hot_pred, alphas if alphas is not None else many_hot_pred

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

                if self.evaluate_on_middle_frames_only:
                    x = int(one_hot_gt.shape[1] * 0.25)
                    one_hot_gt_list.extend(one_hot_gt.numpy()[:, x:-x].flatten().tolist())
                    one_hot_pred_list.extend(logit_fc(one_hot_pred).numpy()[:, x:-x].flatten().tolist())
                else:
                    one_hot_gt_list.extend(one_hot_gt.numpy().flatten().tolist())
                    one_hot_pred_list.extend(logit_fc(one_hot_pred).numpy().flatten().tolist())

                print("\r", i.numpy(), end="")
                if i != 0 or save_visualization_to is None:
                    continue

                with self.summary_writer.as_default():
                    visualizations = visualization_utils.visualize_predictions(
                        frame_sequence.numpy()[:, :, :, :, :3], logit_fc(one_hot_pred).numpy(), one_hot_gt.numpy(),
                        logit_fc(many_hot_pred).numpy() if many_hot_pred is not None else None, many_hot_gt.numpy())
                    tf.summary.image("test/{}/visualization".format(ds_name), visualizations, step=epoch_no)

                    for idx, img in enumerate(visualizations):
                        Image.fromarray(img).save("{}_{}_{:02d}.png".format(save_visualization_to, ds_name, idx))

            with self.summary_writer.as_default():
                for loss_name, loss in self.mean_metrics.items():
                    tf.summary.scalar("test/{}/{}".format(ds_name, loss_name), loss.result(), step=epoch_no)

                f1 = metrics_utils.create_scene_based_summaries(one_hot_pred_list, one_hot_gt_list,
                                                                prefix="test/" + ds_name, step=epoch_no)
                if self.results.get(ds_name, 0) < f1:
                    self.results[ds_name] = f1

    def finish(self):
        with self.summary_writer.as_default():
            for ds_name, f1 in self.results.items():
                tf.summary.scalar("test/" + ds_name + "/scene/best_f1", f1, step=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train TransNet")
    parser.add_argument("config", help="path to config")
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    options = get_options_dict()

    trn_ds = input_processing.train_pipeline(options["trn_files"]) if len(options["trn_files"]) > 0 else None
    if options["transition_only_trn_files"] is not None:
        trn_ds_ = input_processing.train_transition_pipeline(options["transition_only_trn_files"])
        if trn_ds is not None:
            frac = options["transition_only_data_fraction"]
            trn_ds = tf.data.experimental.sample_from_datasets([trn_ds, trn_ds_], weights=[1 - frac, frac])
        else:
            trn_ds = trn_ds_

    tst_ds = [(name, input_processing.test_pipeline(files))
              for name, files in options["tst_files"].items()]

    if options["original_transnet"]:
        net = models.OriginalTransNet()
        logit_fc = lambda x: tf.nn.softmax(x)[:, :, 1]
    elif options["c3d_net"]:
        net = models.C3DNet()
        logit_fc = tf.sigmoid
    else:
        net = transnet.TransNetV2()
        logit_fc = tf.sigmoid
        if options["bi_tempered_loss"]:
            logit_fc = lambda x: tempered_sigmoid(x, t=options["bi_tempered_loss_temp2"])

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

        if options["learning_rate_schedule"] is not None:
            if epoch in options["learning_rate_schedule"]:
                trainer.optimizer.learning_rate = \
                    trainer.optimizer.learning_rate.numpy() * options["learning_rate_decay"]
    trainer.finish()
