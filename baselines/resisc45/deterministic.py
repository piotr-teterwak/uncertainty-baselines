"""
TODO
"""

import os, sys
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins.hparams import api as hp
import numpy as np

from classification_models.tfkeras import Classifiers
import uncertainty_baselines as ub

def get_n_cores():
    """The NSLOTS variable, If NSLOTS is not defined throw an exception."""
    nslots = os.getenv("NSLOTS")
    if nslots is not None:
        return int(nslots)
    raise ValueError("Environment variable NSLOTS is not defined.")

tf.config.set_soft_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(get_n_cores())

DEFAULT_TRAIN_BATCH_SIZE = 512
DEFAULT_NUM_EPOCHS = 300
TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/resisc45/deterministic',
    'The directory where the model weights and training/evaluation summaries '
    'are stored. If you aim to use these as trained models for ensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')

# Learning rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 0.01, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.1, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('l2', 1E-2, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('train_batch_size', DEFAULT_TRAIN_BATCH_SIZE,
                     'The per-core training batch size.')
flags.DEFINE_integer('eval_batch_size', 128,
                     'The per-core validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')


# flags.DEFINE_bool('resnet', False, 'Whether to use ResNet')
flags.DEFINE_bool('pretrained_resnet', False, 'Whether to use Pretrained ResNet')
flags.DEFINE_string('resnet_depth','18','Depth of resnet')
flags.DEFINE_float('reset_prob', 0.0, 'Whether to reset weights with some probability')
flags.DEFINE_integer('reset_stage', None, 'Which block to reset')
flags.DEFINE_integer('final_layer_reset', None, 'Resetting final layers')
flags.DEFINE_integer('unit_2_reset', None, 'How many final units to reset.')

FLAGS = flags.FLAGS

def reinitialize_model(model, probability=0.0, reset_stage_idx=None, final_layer_reset=None, unit_2_reset=None):
  if FLAGS.resnet_depth == '18':
    residual_blocks = ['stage1_unit1','stage1_unit2','stage2_unit1','stage2_unit2','stage3_unit1','stage3_unit2','stage4_unit1','stage4_unit2' ]
  elif FLAGS.resnet_depth in ['34', '50']:
    config = [3,4,6,3]
    residual_blocks = []
    for i, num in enumerate(config):
        for j in range(num):
            residual_blocks.append('stage{}_unit{}'.format(i+1, j+1))
  else:
    raise ValueError('Incorrect resnet depth')
  if reset_stage_idx is not None:
      residual_blocks = residual_blocks[reset_stage_idx:reset_stage_idx + 1]
      print('---> reset_blocks:', residual_blocks, flush=True)
  if final_layer_reset is not None:
      residual_blocks = residual_blocks[-1*final_layer_reset:]
  if unit_2_reset is not None:
      residual_blocks =  ['stage1_unit2','stage2_unit2','stage3_unit2','stage4_unit2' ]
      random.shuffle(residual_blocks)
      residual_blocks = residual_blocks[:unit_2_reset]
  initializers = []
  weights = []
  for block in residual_blocks:
    if np.random.random() >= probability:
      pass
    else:
      for layer in model.layers:
        if block in layer.name:
          if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            weights += [layer.kernel]
            initializers += [layer.kernel_initializer]
            if layer.bias is not None:
              weights += [layer.bias]
              initializers += [layer.bias_initializer]
          elif isinstance(layer, tf.keras.layers.BatchNormalization):
            weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
            initializers += [layer.gamma_initializer,
                            layer.beta_initializer,
                            layer.moving_mean_initializer,
                            layer.moving_variance_initializer]
  for w, init in zip(weights, initializers):
    w.assign(init(w.shape, dtype=w.dtype))


def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg

  logging.info(FLAGS.flag_values_dict())
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  physical_devices = tf.config.list_physical_devices('GPU')
  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
  except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

  data_dir = FLAGS.data_dir
  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  # Initialize distribution strategy on flag-specified accelerator
  use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  train_batch_size = FLAGS.train_batch_size * FLAGS.num_cores
  eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores

  data_dir = FLAGS.data_dir

  dataset_train_builder = ub.datasets.get(
      'resisc45', split='train', data_dir=data_dir, download_data=True)
  dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

  dataset_validation_builder = ub.datasets.get(
      'resisc45',
      split='validation',
      data_dir=data_dir,
      is_training=not FLAGS.use_validation,
      download_data=True)
  validation_batch_size = (
      eval_batch_size if FLAGS.use_validation else train_batch_size)
  dataset_validation = dataset_validation_builder.load(
      batch_size=validation_batch_size)
  if FLAGS.use_validation:
    dataset_validation = strategy.experimental_distribute_dataset(
        dataset_validation)
  else:
    # Note that this will not create any mixed batches of train and validation
    # images.
    dataset_train = dataset_train.concatenate(dataset_validation)

  dataset_train = strategy.experimental_distribute_dataset(dataset_train)

  dataset_test_builder = ub.datasets.get(
      'resisc45', split='test', data_dir=data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
  dataset_test = strategy.experimental_distribute_dataset(dataset_test)

  ds_info = tfds.builder('resisc45', data_dir=FLAGS.data_dir).info
  num_examples = ds_info.splits["train"].num_examples
  steps_per_epoch = (num_examples * TRAIN_SPLIT_PERCENT // 100 ) // train_batch_size
  steps_per_validation_eval = (num_examples * VALIDATION_SPLIT_PERCENT // 100) // eval_batch_size
  steps_per_test_eval = (num_examples * TEST_SPLIT_PERCENT // 100) // eval_batch_size
  num_classes = 45

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 deterministic model.')

    if FLAGS.pretrained_resnet:
      ResNet, preprocess_input = Classifiers.get('resnet{}'.format(FLAGS.resnet_depth))
      inputs = tf.keras.layers.Input((None,None,3))
      rescaled_inputs = inputs * 255.
      resnet_trunk = ResNet((None, None, 3), l2=FLAGS.l2, weights='imagenet', include_top=False)

      if FLAGS.reset_prob > 0.0:
          reinitialize_model(resnet_trunk, probability = FLAGS.reset_prob, reset_stage_idx=FLAGS.reset_stage, final_layer_reset=FLAGS.final_layer_reset, unit_2_reset=FLAGS.unit_2_reset)
      resnet_trunk_out = resnet_trunk(rescaled_inputs)
      pooled_resnet = tf.keras.layers.GlobalAveragePooling2D()(resnet_trunk_out)
      output = tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
              kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2),
              bias_regularizer=tf.keras.regularizers.l2(FLAGS.l2))(pooled_resnet)
      model = tf.keras.Model(inputs=inputs, outputs=output,name='pretrained_resnet')
    # else:
    #   model = resnet50_deterministic(
    #       input_shape=utils.load_input_shape(dataset_train),
    #       num_classes=num_classes)
    
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    base_lr = FLAGS.base_learning_rate * train_batch_size / DEFAULT_TRAIN_BATCH_SIZE
    lr_decay_epochs = [
        (int(start_epoch_str) * FLAGS.train_epochs) // DEFAULT_NUM_EPOCHS
        for start_epoch_str in FLAGS.lr_decay_epochs
    ]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)
    metrics = {
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.CategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.CategoricalAccuracy(),
        'train/l2':
            tf.keras.metrics.Mean()
    }
    if FLAGS.use_validation:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy': tf.keras.metrics.CategoricalAccuracy(),
      })
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope()
      # so that optimizer slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  @tf.function
  def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']
      
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      probs = tf.nn.softmax(logits)

      metrics['train/l2'].update_state(l2_loss)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for i in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, num_steps):
    """Evaluation step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      
      negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
              y_true=labels,
              y_pred=logits,
              from_logits=True))
      probs = tf.nn.softmax(logits)

      metrics[dataset_split + '/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics[dataset_split + '/accuracy'].update_state(labels, probs)

    for i in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'train/ms_per_example': tf.keras.metrics.Mean()})
  start_time = time.time()
  
  train_iterator = iter(dataset_train)
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    # if tb_callback:
    #   tb_callback.on_epoch_begin(epoch)
    train_start_time = time.time()
    train_step(train_iterator)
    ms_per_example = (time.time() - train_start_time) * 1e6 / train_batch_size
    metrics['train/ms_per_example'].update_state(ms_per_example)

    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * FLAGS.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   FLAGS.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)
    # if tb_callback:
    #   tb_callback.on_epoch_end(epoch)

    if FLAGS.use_validation:
      validation_iterator = iter(dataset_validation)
      test_step(
          validation_iterator, 'validation', steps_per_validation_eval)

    test_iterator = iter(dataset_test)
    logging.info('Starting to run eval at epoch: %s', epoch)
    test_start_time = time.time()
    test_step(test_iterator, 'test', steps_per_test_eval)
    ms_per_example = (time.time() - test_start_time) * 1e6 / eval_batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%, L2: %.4f',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100, metrics['train/l2'].result())
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
