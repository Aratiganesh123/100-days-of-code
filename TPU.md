
## TPU-speed data pipelines: tf.data.Dataset and TFRecords


- TPUs are hardware accelerators specialized in deep learning tasks.
- Cloud TPUs are available in a base configuration with 8 cores and also in larger configurations called "TPU pods" of up to 2048 cores
- A code snippet needs to be written to detect TPU
- Typically used where large matrix multiplication tasks dominate

```
try: # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError: # detect GPUs
    strategy = tf.distribute.MirroredStrategy() # for CPU/GPU or multi-GPU machines

# use TPUStrategy scope to define model
with strategy.scope():
  model = tf.keras.Sequential( ... )
  model.compile( ... )

# train model normally on a tf.data.Dataset
model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=...)

```
### Components of a TPU

- MXU : which runs matrix multiplications. Mixed precision 16-32 bit floating point format.
- VPU : other tasks such as activations, softmax, etc. The VPU handles float32 and int32 computations

### TF-Records
- Loading data one by one takes time
- Feed data to the TPU fast enough to keep them busy
- This will lead to a lot of connection cost back and forth
- Batch them and use tf.data.Dataset

- Rule of thumb for optimal GCS throughput : split your data across several (10s to 100s) larg-ish files (10s to 100s of MB)

For optimal performance, it is recommended to use the following more complex code to read from multiple TFRecord files at once. This code will read from N files in parallel and disregard data order in favor of reading speed.

```
AUTOTUNE = tf.data.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

filenames = tf.io.gfile.glob(FILENAME_PATTERN)
dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
dataset = dataset.with_options(ignore_order)
dataset = dataset.map(...) # do the TFRecord decoding here - see below
```

Reading from TFRecords

```
def read_tfrecord(data):
  features = {
    # tf.string = byte string (not text string)
    "image": tf.io.FixedLenFeature([], tf.string), # shape [] means scalar, here, a single byte string
    "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar, i.e. a single item
    "size": tf.io.FixedLenFeature([2], tf.int64),  # two integers
    "float_data": tf.io.VarLenFeature(tf.float32)  # a variable number of floats
  }

  # decode the TFRecord
  tf_record = tf.io.parse_single_example(data, features)

  # FixedLenFeature fields are now ready to use
  sz = tf_record['size']

  # Typical code for decoding compressed images
  image = tf.io.decode_jpeg(tf_record['image'], channels=3)

  # VarLenFeature fields require additional sparse.to_dense decoding
  float_data = tf.sparse.to_dense(tf_record['float_data'])

  return image, sz, float_data

# decoding a tf.data.TFRecordDataset
dataset = dataset.map(read_tfrecord)
# now a dataset of triplets (image, sz, float_data)
```