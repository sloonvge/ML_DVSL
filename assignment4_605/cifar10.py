import numpy as np 
import os
import tensorflow as tf

#cifar10 数据读取
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
def read_cifar10(filenames, buffer_size, batch_size, seed, shuffle):
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    dataset = tf.data.FixedLengthRecordDataset(filenames=filenames, record_bytes=record_bytes)

    dataset = dataset.map(lambda x : tf.decode_raw(x, tf.uint8)) # byte to tensor
        
    result.label = dataset.map(lambda x : tf.cast(
        tf.strided_slice(x, [0], [label_bytes]), 
        tf.int32
    )).map(lambda x: tf.one_hot(x, 10, 1, 0)) # label to one_hot
        
    result.uint8image = dataset.map(lambda x : tf.reshape(
        tf.strided_slice(x, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )).map(
        lambda x: tf.transpose(x, [1, 2, 0])).map(lambda x: (x - tf.reduce_mean(x, 2, keepdims=True))).map(
        lambda x: tf.cast(x, tf.float32)) # image: [C H W] to [H W C]
        
    dataset = tf.data.Dataset.zip((result.uint8image, result.label))
        
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    return dataset

    
def load_cifar10(data_dir, batch_size, test=False, shuffle=True, seed=None):
    if not test:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
        epoch_example = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        epoch_example = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    min_after_dequeue = int(epoch_example * 0.4)
    return read_cifar10(filenames,
                 buffer_size=min_after_dequeue + 3 * batch_size,
                 batch_size=batch_size,
                 seed=seed,
                 shuffle=shuffle)