# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implementation of translation consistency regularization.
"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from libml import utils, data, models
from libml.utils import EasyDict
from libml import ctaugment

FLAGS = flags.FLAGS


class TranslationConsistencyRegularization(models.MultiModel):
    def augment(self, x, tcr_augment, **kwargs):
        del kwargs
        for augmentation in tcr_augment:
            # if augmentation == "rotate":
            #     # Taken from ctaugment.
            #     rotate
        return x



    def model(self, batch, lr, wd, ema, warmup_pos, consistency_weight, tcr_augment, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(tf.float32, [batch] + hwc, 'xt')  # For training
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [batch, 2] + hwc, 'y') # The unlabeled data
        l_in = tf.placeholder(tf.int32, [batch], 'labels')
        l = tf.one_hot(l_in, self.nclass)

        warmup = tf.clip_by_value(tf.to_float(self.step) / (warmup_pos * (FLAGS.train_kimg << 10)), 0, 1)
        lrate = tf.clip_by_value(tf.to_float(self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Labeled data.
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits_x = classifier(xt_in, training=True)
        post_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Take only first call to update batch norm.
        
        # Unlabeled data.
        classifier_embedding = lambda x, **kw: self.classifier(x, **kw, **kwargs).embeds
        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        y_delta = self.augment(y, augment=tcr_augment) # Apply tcr_augment
        y_1, y_2 = tf.split(y, 2)
        y_1_delta, y_2_delta = tf.split(y_delta, 2)
        embeds_y_1 = classifier_embedding(y_1, training=True)
        embeds_y_1_delta = classifier_embedding(y_1_delta, training=True)
        embeds_y_2 = classifier_embedding(y_2, training=True)
        embeds_y_2_delta = classifier_embedding(y_2_delta, training=True)
        loss_tcr = tf.losses.mean_squared_error((y_1_delta - y_1) - (y_2_delta - y_2))
        loss_tcr = tf.reduce_mean(loss_tcr)
        tf.summary.scalar('losses/xeu', loss_tcr)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=l, logits=logits_x)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar('losses/xe', loss)


        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v) for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss + loss_tcr * warmup * consistency_weight + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = TranslationConsistencyRegularization(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        warmup_pos=FLAGS.warmup_pos,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,
        smoothing=FLAGS.smoothing,
        consistency_weight=FLAGS.consistency_weight,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('consistency_weight', 1., 'Consistency weight.')
    flags.DEFINE_float('warmup_pos', 0.4, 'Relative position at which constraint loss warmup ends.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('smoothing', 0.1, 'Label smoothing.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    FLAGS.set_default('dataset', 'cifar10.3@250-5000')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)
    # First d is for labeled samples, second d for weakly augmented, third d for strongly augmented.
    # See github repo for meanings. d is for default. Use d.x.x for default augmentation for labeled samples
    # and no augmentation for unlabeled samples. tcr_augment is applied on top of unlabeled samples.
    FLAGS.set_default('augment', 'd.d.d') 
    # List of augment functions separated by . on top of existing augment. Currently only rotate is supported.
    FLAGS.set_default('tcr_augment', 'rotate') 
    app.run(main)