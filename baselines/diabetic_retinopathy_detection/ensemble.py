# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Ensemble of ResNet50 models on Kaggle's Diabetic Retinopathy Detection dataset.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py`
over different seeds.
"""

import os, sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import

from classification_models.tfkeras import Classifiers
import uncertainty_baselines as ub

# Data load / output flags.
flags.DEFINE_string(
    'checkpoint_dir', '/tmp/diabetic_retinopathy_detection/deterministic',
    'The directory from which the trained deterministic '
    'model weights are retrieved.')
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/ensemble',
    'The directory where the ensemble model weights '
    'and training/evaluation summaries are stored.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_batch_size', 128,
                     'The per-core validation/test batch size.')
flags.DEFINE_float('l2', 1E-3, 'L2 regularization coefficient.') # 1E-1

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU, otherwise CPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer(
    'num_cores', 1,
    'Number of TPU cores or number of GPUs; only support 1 GPU for now.')

flags.DEFINE_bool('resnet', False, 'Whether to use ResNet')
flags.DEFINE_string('resnet_depth','18','Depth of resnet')

flags.DEFINE_bool('single_model', False, 'Whether only testing single model')

flags.DEFINE_string('reset_stage_1', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_2', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_3', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_4', None, 'Which block to reset')
  
FLAGS = flags.FLAGS


def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)

  if FLAGS.single_model:
    for path, _, files in tf.io.gfile.walk(checkpoint_dir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(latest_checkpoint_without_suffix)
        break
    return paths

  subdirectories = tf.io.gfile.glob(os.path.join(checkpoint_dir, '*'))
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
        break
  return paths


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving Deep Ensemble predictions to %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')

  if FLAGS.use_gpu:
    logging.info('Use GPU')
  else:
    logging.info('Use CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  # As per the Kaggle challenge, we have split sizes:
  # train: 35,126
  # validation: 10,906 (currently unused)
  # test: 42,670
  ds_info = tfds.builder('diabetic_retinopathy_detection/btgraham-300', data_dir=FLAGS.data_dir).info
  eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores
  steps_per_eval = ds_info.splits['test'].num_examples // eval_batch_size

  dataset_test_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='test', data_dir=FLAGS.data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  if FLAGS.resnet:
    ResNet, preprocess_input = Classifiers.get('resnet{}'.format(FLAGS.resnet_depth))
    inputs = tf.keras.layers.Input((None,None,3))
    rescaled_inputs = inputs * 255.
    resnet_trunk = ResNet((None, None, 3), l2=FLAGS.l2, weights='imagenet', include_top=False)(rescaled_inputs)
    
    pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk)
    num_classes = 5
    output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
          kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2),
          bias_regularizer=tf.keras.regularizers.l2(FLAGS.l2))(pooled_resnet)
    model = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_{}'.format(FLAGS.resnet_depth))
  else:
    model = ub.models.wide_resnet(
      input_shape=ds_info.features['image'].shape,
      depth=28,
      width_multiplier=10,
      num_classes=num_classes,
      l2=0.,
      version=2)
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  # Search for checkpoints from their index file; then remove the index suffix.
  ensemble_filenames = parse_checkpoint_dir(FLAGS.checkpoint_dir)
  ensemble_size = len(ensemble_filenames)
  logging.info('Ensemble size: %s', ensemble_size)
  logging.info('Ensemble number of weights: %s',
               ensemble_size * model.count_params())
  logging.info('Ensemble filenames: %s', str(ensemble_filenames))
  checkpoint = tf.train.Checkpoint(model=model)


  # Write model predictions to files.
  for member, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename).expect_partial()
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    filename = f'{member}.npy'
    filename = os.path.join(FLAGS.output_dir, filename)
    if not tf.io.gfile.exists(filename):
      logits = []
      test_iterator = iter(dataset_test)
      for i in range(steps_per_eval):
        inputs = next(test_iterator)  # pytype: disable=attribute-error
        images = inputs['features']
        logits.append(model(images, training=False))

        if i % 100 == 0:
          logging.info(
              'Ensemble member %d/%d: Completed %d of %d eval steps.',
              member + 1,
              ensemble_size,
              i + 1,
              steps_per_eval)

      logits = tf.concat(logits, axis=0)
      with tf.io.gfile.GFile(filename, 'w') as f:
        np.save(f, logits.numpy())

    percent = (member + 1) / ensemble_size
    message = (
        '{:.1%} completion for prediction: ensemble member {:d}/{:d}.'.format(
            percent, member + 1, ensemble_size))
    logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
      'test/diversity': rm.metrics.AveragePairwiseDiversity(normalize_disagreement=False),
  }

  for i in range(ensemble_size):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.CategoricalAccuracy())

  # Evaluate model predictions.
  logits_dataset = []
  for member in range(ensemble_size):
    filename = f'{member}.npy'
    filename = os.path.join(FLAGS.output_dir, filename)
    with tf.io.gfile.GFile(filename, 'rb') as f:
      logits_dataset.append(np.load(f))

  logits_dataset = tf.convert_to_tensor(logits_dataset)
  test_iterator = iter(dataset_test)

  for step in range(steps_per_eval):
    inputs = next(test_iterator)  # pytype: disable=attribute-error
    labels = inputs['labels']
    logits = logits_dataset[:, (step * eval_batch_size):((step + 1) *
                                                         eval_batch_size)]
    int_labels = tf.argmax(labels, axis=1)
    negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy()
    negative_log_likelihood_metric.add_batch(
        logits, labels=int_labels)
    negative_log_likelihood = list(
        negative_log_likelihood_metric.result().values())[0]
    per_probs = tf.nn.softmax(logits)
    probs = tf.reduce_mean(per_probs, axis=0)
    gibbs_ce_metric = rm.metrics.GibbsCrossEntropy()
    gibbs_ce_metric.add_batch(logits, labels=int_labels)
    gibbs_ce = list(gibbs_ce_metric.result().values())[0]
    metrics['test/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['test/gibbs_cross_entropy'].update_state(gibbs_ce)
    metrics['test/accuracy'].update_state(labels, probs)
    metrics['test/ece'].add_batch(probs, label=labels)
    metrics['test/diversity'].add_batch(per_probs)

    for i in range(ensemble_size):
      member_probs = per_probs[i]
      member_loss = tf.keras.losses.categorical_crossentropy(labels, member_probs)
      metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
      metrics['test/accuracy_member_{}'.format(i)].update_state(
          labels, member_probs)

  total_results = {name: metric.result() for name, metric in metrics.items()}
  # Results from Robustness Metrics themselves return a dict, so flatten them.
  total_results = utils.flatten_dictionary(total_results)
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
