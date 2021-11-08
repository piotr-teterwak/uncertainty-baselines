"""resisc45 dataset builder."""

from typing import Dict, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

class Resisc45Dataset(base.BaseDataset):
  """resisc45 dataset builder class."""

  def __init__(
      self,
      split: str,
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = tf.data.experimental.AUTOTUNE,
      data_dir: Optional[str] = None,
      download_data: bool = False,
      is_training: Optional[bool] = None):
    """Create a resisc45 tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      download_data: Whether or not to download data before loading.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    dataset_builder = tfds.builder(
        'resisc45', data_dir=data_dir)

    num_examples = dataset_builder.info.splits["train"].num_examples
    train_count = num_examples * TRAIN_SPLIT_PERCENT // 100
    val_count = num_examples * VALIDATION_SPLIT_PERCENT // 100
    test_count = num_examples * TEST_SPLIT_PERCENT // 100
    
    super().__init__(
        name='resisc45',
        dataset_builder=dataset_builder,
        split=self._get_split(split, train_count, val_count, test_count),
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _get_split(self, split_name, train_count, val_count, test_count):
    tfds_splits = {
        "train":
            "train[:{}]".format(train_count),
        "validation":
            "train[{}:{}]".format(train_count, train_count + val_count),
        "trainval":
            "train[:{}]".format(train_count + val_count),
        "test":
            "train[{}:]".format(train_count + val_count),
        "train800":
            "train[:800]",
        "val200":
            "train[{}:{}]".format(train_count, train_count+200),
        "train800val200":
            "train[:800]+train[{}:{}]".format(train_count, train_count+200),
    }

    # # Creates a dict with example counts for each split.
    # num_samples_splits = {
    #     "train": train_count,
    #     "val": val_count,
    #     "trainval": train_count + val_count,
    #     "test": test_count,
    #     "train800": 800,
    #     "val200": 200,
    #     "train800val200": 1000,
    # }

    return tfds_splits[split_name]


  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)

      if self._is_training:

          # Inception Crop
          width = tf.cast(tf.shape(image),tf.float32)[1]
          height = tf.cast(tf.shape(image),tf.float32)[0]
          min_side = tf.minimum(width,height)

          crop_size =  tf.random.uniform([], minval=0.08, maxval=1.) * tf.cast(min_side, tf.float32)
          crop_size = tf.cast(crop_size, tf.int32)
          image = tf.image.random_crop(image, [crop_size,crop_size,3])
          image = tf.image.resize(image, [224,224])
          image = tf.image.random_flip_left_right(image)

      else:
          offset_width = (256 - 224) // 2
          offset_height = (256 - 224) // 2
          image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, 224,224)

      label = tf.one_hot(example['label'], 45, dtype=tf.float32)
      
      parsed_example = {
          'features': image,
          'labels': label,
          'name': example['filename'],
      }
      return parsed_example

    return _example_parser