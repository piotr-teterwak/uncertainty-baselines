"""Ensemble on Resisc45.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py` over different
seeds.
"""

import os, sys

from absl import app
from absl import flags
from absl import logging

import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.insert(0, os.path.abspath('./'))
from classification_models.classification_models.tfkeras import Classifiers
import uncertainty_baselines as ub

TEST_SPLIT_PERCENT = 20

flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.')
flags.mark_flag_as_required('checkpoint_dir')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'output_dir', '/tmp/resisc45/deterministic',
    'The directory where the model weights and training/evaluation summaries '
    'are stored. If you aim to use these as trained models for ensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 128,
                     'The per-core validation/test batch size.')
flags.DEFINE_float('l2', 1E-2, 'L2 regularization coefficient.')

flags.DEFINE_bool('resnet', False, 'Whether to use ResNet')
flags.DEFINE_string('resnet_depth','18','Depth of resnet')

flags.DEFINE_bool('single_model', False, 'Whether only testing single model')

flags.DEFINE_string('reset_stage_1', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_2', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_3', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_4', None, 'Which block to reset')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')

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


def flatten_dictionary(x):
  """Flattens a dictionary where elements may itself be a dictionary.

  This function is helpful when using a collection of metrics, some of which
  include Robustness Metrics' metrics. Each metric in Robustness Metrics
  returns a dictionary with potentially multiple elements. This function
  flattens the dictionary of dictionaries.

  Args:
    x: Dictionary where keys are strings such as the name of each metric.

  Returns:
    Flattened dictionary.
  """
  outputs = {}
  for k, v in x.items():
    if isinstance(v, dict):
      if len(v.values()) == 1:
        # Collapse metric results like ECE's with dicts of len 1 into the
        # original key.
        outputs[k] = list(v.values())[0]
      else:
        # Flatten metric results like diversity's.
        for v_k, v_v in v.items():
          outputs[f'{k}/{v_k}'] = v_v
    else:
      outputs[k] = v
  return outputs


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

  dataset = ub.datasets.get(
      'resisc45', split='test', data_dir=FLAGS.data_dir).load(batch_size=batch_size)
  test_datasets = {'resisc45': dataset}

  ds_info = tfds.builder('resisc45', data_dir=FLAGS.data_dir).info
  num_examples = ds_info.splits["train"].num_examples
  steps_per_eval = (num_examples * TEST_SPLIT_PERCENT // 100) // batch_size
  num_classes = 45

  if FLAGS.resnet:
    ResNet, preprocess_input = Classifiers.get('resnet{}'.format(FLAGS.resnet_depth))
    inputs = tf.keras.layers.Input((None,None,3))
    rescaled_inputs = inputs * 255.
    resnet_trunk = ResNet((None, None, 3), l2=FLAGS.l2, weights='imagenet', include_top=False)(rescaled_inputs)
    
    pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk)
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
  num_datasets = len(test_datasets)
  for m, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename).expect_partial()
    for n, (name, test_dataset) in enumerate(test_datasets.items()):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      if not tf.io.gfile.exists(filename):
        logits = []
        test_iterator = iter(test_dataset)
        for _ in range(steps_per_eval):
          features = next(test_iterator)['features']  # pytype: disable=unsupported-operands
          logits.append(model(features, training=False))

        logits = tf.concat(logits, axis=0)
        with tf.io.gfile.GFile(filename, 'w') as f:
          np.save(f, logits.numpy())
      percent = (m * num_datasets + (n + 1)) / (ensemble_size * num_datasets)
      message = ('{:.1%} completion for prediction: ensemble member {:d}/{:d}. '
                 'Dataset {:d}/{:d}'.format(percent,
                                            m + 1,
                                            ensemble_size,
                                            n + 1,
                                            num_datasets))
      logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
      'test/diversity': rm.metrics.AveragePairwiseDiversity(normalize_disagreement=False),
  }

  for i in range(ensemble_size):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.CategoricalAccuracy())

  # Evaluate model predictions.
  for n, (name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for m in range(ensemble_size):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      labels = next(test_iterator)['labels']  # pytype: disable=unsupported-operands
      logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
      int_labels = tf.argmax(labels, axis=1)
      negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy()
      negative_log_likelihood_metric.add_batch(logits, labels=int_labels)
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
      metrics['test/ece'].add_batch(probs, label=int_labels)

      for i in range(ensemble_size):
        member_probs = per_probs[i]
        member_loss = tf.keras.losses.categorical_crossentropy(
                labels, member_probs)
        metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
        metrics['test/accuracy_member_{}'.format(i)].update_state(
                labels, member_probs)
      metrics['test/diversity'].add_batch(per_probs)

    message = ('{:.1%} completion for evaluation: dataset {:d}/{:d}'.format(
        (n + 1) / num_datasets, n + 1, num_datasets))
    logging.info(message)

  total_results = {name: metric.result() for name, metric in metrics.items()}
  # Results from Robustness Metrics themselves return a dict, so flatten them.
  total_results = flatten_dictionary(total_results)
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
