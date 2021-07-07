import os
import tensorflow as tf

class decode_fn:
        def __init__(self,is_training, data_shape):
            pass
        
        def __call__(self, record_bytes):
            example = tf.io.parse_single_example(
                # Data
                record_bytes,

                # Schema
                {"inputs": tf.io.FixedLenFeature([], dtype=tf.string),
                 "labels": tf.io.FixedLenFeature([], dtype=tf.string)}
            )
            
            inputs = tf.io.parse_tensor(example["inputs"],
                               out_type = tf.float32)
            
            labels = tf.io.parse_tensor(example["labels"],
                               out_type = tf.float32)
            
            return inputs, labels


def TFR_load(
    path,
    BATCH_SIZE,
    NUM_TRAIN_DATA,
    is_training = False,
    data_shape = [5]
):
    
    files = [path+n for n in os.listdir(path)]
    ds = tf.data.TFRecordDataset(files)\
                    .map(decode_fn(is_training, data_shape))\
                    .shuffle(NUM_TRAIN_DATA, reshuffle_each_iteration=True)\
                    .batch(BATCH_SIZE)\
                    .prefetch(tf.data.AUTOTUNE)
                    #.repeat()\
                    
                    
    return ds