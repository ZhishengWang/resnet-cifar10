# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys

import cifar_input_v2
import numpy as np
#import resnet_model_expert
import resnet_model
import tensorflow as tf
import cPickle

log_root = '/home/zswang/training_logs/resnet_model'
log_expert_root = '/home/zswang/training_logs/resnet_model'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')
tf.app.flags.DEFINE_string('train_data_path', 'cifar10/data_batch*',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', 'cifar10/test_batch.bin',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', log_root + '/train',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', log_root + '/test',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', log_root,
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_string('log_root_expert', log_expert_root,
                           'Directory to keep the expert checkpoints..')
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_string('train_type', 'expert',
                           'training types.')


def train(hps):
  """Training loop."""
  images, labels, prob = cifar_input_v2.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, prob, FLAGS.mode)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))
  
  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'mask_loss': model.mask_loss,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.01

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results

      if train_step < 30000:
        self._lrn_rate = 0.01
      elif train_step < 60000:
        self._lrn_rate = 0.001
      elif train_step < 80000:
        self._lrn_rate = 0.0001
      else:
        self._lrn_rate = 0.00001
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.92)
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)

def evaluate(hps):
  """Eval loop."""
  images, labels, prob = cifar_input_v2.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, prob, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50) #, gpu_options=gpu_options
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, mask_loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.mask_loss, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f,mask_loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, mask_loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


def gene_prob(hps):
  """Generating loop."""
  images, labels, prob = cifar_input_v2.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, 'eval')
  model = resnet_model.ResNet(hps, images, labels, prob, 'eval')
  model.build_graph()
  saver = tf.train.Saver()

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root_expert)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root_expert)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    obj_ = []
    i = 1
    count = 0
    f = open('prob.pkl','wb')
    for _ in six.moves.range(40000/hps.batch_size):
      (predictions, truth) = sess.run(
          [model.predictions,
           model.labels])

      pred_probability = np.sum(truth*predictions,axis=1) #with shape [128]
      if pred_probability[0] >= 0.90:
        count = count + 1
      obj_.append(pred_probability)
      
      if i%100 == 0:
          print i,'----->',pred_probability
      i = i + 1
    print 'the ratio is:', 1.0*count/40000
    cPickle.dump(obj=obj_, file=f, protocol=0)
    f.close()
    #create bin file
    """
    file = open('prob.pkl','rb')
    data = cPickle.load(file)
    arr = np.array(data)
    arr.tofile("prob_pkl.bin")
    """
    break

def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 100
  elif FLAGS.mode == 'gene':
    batch_size = 1

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100
  
  hps = resnet_model.HParams(batch_size=batch_size,
                         num_classes=num_classes,
                         min_lrn_rate=0.0001,
                         lrn_rate=0.01,
                         num_residual_units=5,
                         use_bottleneck=False,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1,
                         optimizer='mom')
  
  

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)
    elif FLAGS.mode == 'gene':
      gene_prob(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
