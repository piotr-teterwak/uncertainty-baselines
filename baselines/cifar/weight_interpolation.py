import os

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

flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.')
flags.mark_flag_as_required('checkpoint_dir')
flags.DEFINE_string('resnet_depth','18','Depth of resnet')

flags.DEFINE_string('reset_stage_1', None, 'Which block to reset')
flags.DEFINE_string('reset_stage_2', None, 'Which block to reset')

FLAGS = flags.FLAGS


def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)

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
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

  dataset = ub.datasets.get(
      FLAGS.dataset,
      data_dir=FLAGS.data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST).load(batch_size=batch_size)
  test_datasets = {'clean': dataset}

  ds_info = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir).info
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  ResNet, preprocess_input = Classifiers.get('resnet{}'.format(FLAGS.resnet_depth))
  inputs = tf.keras.layers.Input((32,32,3))
  reshaped_inputs = tf.image.resize(inputs, [224,224], method='bicubic')
  rescaled_inputs = reshaped_inputs * 255.
  resnet_trunk = ResNet((224, 224, 3), l2=FLAGS.l2, weights=None, include_top=False)(rescaled_inputs)
  pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk)
  output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
        kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2),
        bias_regularizer=tf.keras.regularizers.l2(FLAGS.l2))(pooled_resnet)
  model_1 = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_{}'.format(FLAGS.resnet_depth))
  model_2 = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_{}'.format(FLAGS.resnet_depth))
  model = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet_{}'.format(FLAGS.resnet_depth))


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
  checkpoint = tf.train.Checkpoint(model=model_1)
  checkpoint_2 = tf.train.Checkpoint(model=model_2)

  num_inters = 11
  checkpoint.restore(ensemble_filenames[0]).expect_partial()
  weights_1 = model_1.get_weights()
  checkpoint_2.restore(ensemble_filenames[1]).expect_partial()
  weights_2 = model_2.get_weights()

  inter_weights = np.linspace(0, 1, num_inters)
  for m in range(num_inters):
    updated_weights = []
    for i in range(len(weights_2)):
      updated_weights.append((1-inter_weights[m])*weights_1[i] + inter_weights[m]*weights_2[i])
    model.set_weights(updated_weights)

    logging.info(model.get_weights()[0])
    logging.info(weights_2[0])
    logging.info(weights_1[0])

# Write model predictions to files.
    num_datasets = len(test_datasets)
    for n, (name, test_dataset) in enumerate(test_datasets.items()):
      filename = 'interpolate_{member}.npy'.format(member=m)
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
      'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
      'test/diversity': rm.metrics.AveragePairwiseDiversity(normalize_disagreement=False),
  }
  
  for i in range(num_inters):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())

  # Evaluate model predictions.
  for n, (name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for m in range(num_inters):
      filename = 'interpolate_{member}.npy'.format(member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      labels = next(test_iterator)['labels']  # pytype: disable=unsupported-operands
      logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
      labels = tf.cast(labels, tf.int32)
      negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy()
      negative_log_likelihood_metric.add_batch(logits, labels=labels)
      negative_log_likelihood = list(
          negative_log_likelihood_metric.result().values())[0]
      per_probs = tf.nn.softmax(logits)
      probs = tf.reduce_mean(per_probs, axis=0)
      if name == 'clean':
        gibbs_ce_metric = rm.metrics.GibbsCrossEntropy()
        gibbs_ce_metric.add_batch(logits, labels=labels)
        gibbs_ce = list(gibbs_ce_metric.result().values())[0]
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/gibbs_cross_entropy'].update_state(gibbs_ce)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)

        for i in range(num_inters):
          member_probs = per_probs[i]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
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
  total_results = utils.flatten_dictionary(total_results)
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
