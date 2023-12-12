# starting from this tutorial: https://jaxopt.github.io/stable/auto_examples/deep_learning/haiku_vae.html
# and this notebook: https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/2.PartII-GenerativeModels.ipynb

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
from jaxopt import OptaxSolver
import numpy as onp
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

from tqdm import tqdm
import pickle
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
import numpy as np

import herculens_util

IMAGE_SHAPE = (101,101)

use_lensing_loss = True

debug = False

tfrecord_path = 'COSMOS_23.5_training_sample/custom_cutouts_round2/training_data.tfrecord'
batch_size = 256
n_epochs = 10
init_file = 'model_test_epoch010.pckl'
loss_path = 'lensing2_loss.csv'

#######################################################
# function to load in the dataset from a .tfrecord file
#######################################################
# FROM PALTAS (cite here)
def generate_tf_dataset(tf_record_path,batch_size,
    n_epochs,shuffle=True):
    """Generate a TFDataset that a model can be trained with.

    Args:
        tf_record_paths (str, or [str,...]) A string specifying the paths to
            the TFRecords that will be used in the dataset. Can also be a list
            of strings for specifying multiple tf_record_paths.
        batch_size (int): The batch size that will be used for training
        n_epochs (int): The number of training epochs. The dataset object will
            deal with iterating over the data for repeated epochs.
        shuffle (bool): if True (default is True), randomly shuffles dataset.

    Returns:
        (tf.Dataset): A tf.Dataset object that returns the input image and the
        output labels.

    Notes:
        Do not use kwargs_detector if noise was already added during dataset
        generation.
    """
    # Read the TFRecords
    raw_dataset = tf.data.TFRecordDataset(tf_record_path)

    # Create the feature decoder that will be used
    def parse_image_features(example):  # pragma: no cover
        data_features = {
            'image': tf.io.FixedLenFeature([],tf.string),
            'height': tf.io.FixedLenFeature([],tf.int64),
            'width': tf.io.FixedLenFeature([],tf.int64),
            'index': tf.io.FixedLenFeature([],tf.int64),
        }
        # Set the log learning params to an empy list if no value is provided.

        parsed_dataset = tf.io.parse_single_example(example,data_features)
        image = tf.io.decode_raw(parsed_dataset['image'],out_type=float)
        image = tf.reshape(image,(parsed_dataset['height'],
            parsed_dataset['width'],1))
        
        return image

    # Select the buffer size to be slightly larger than the batch
    buffer_size = int(batch_size*1.2)

    # Set the feature decoder as the mapping function. Drop the remainder
    # in the case that batch_size does not divide the number of training
    # points exactly
    dataset = raw_dataset.map(parse_image_features,
        num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(
        n_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


class Encoder(hk.Module):
  """Encoder model."""

  def __init__(self, hidden_size=512, latent_size=10):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size

  def __call__(self, x):

    #x = hk.Flatten()(x)
    #x = hk.Linear(self._hidden_size)(x)
    #x = jax.nn.relu(x)

    # Lanusse et al. architecture
    # from: https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/2.PartII-GenerativeModels.ipynb

    # add an axis for channels
    x = x[..., jnp.newaxis]

    x = hk.Conv2D(16, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')

    x = hk.Conv2D(32, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x,  window_shape=3, strides=2, padding='SAME')

    x = hk.Conv2D(64, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')

    x = hk.Conv2D(128, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')

    x = hk.Conv2D(128, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')

    x = hk.Flatten()(x)

    x = hk.Linear(256)(x)
    x = jax.nn.leaky_relu(x)

    mean = hk.Linear(self._latent_size)(x)
    #log_stddev = hk.Linear(self._latent_size)(x)
    #stddev = jnp.exp(log_stddev)
    stddev = jax.nn.softplus(hk.Linear(self._latent_size)(x)) + 1e-3

    return mean, stddev


class Decoder(hk.Module):
  """Decoder model."""

  def __init__(self, hidden_size=512, output_shape=IMAGE_SHAPE):
    super().__init__()
    self._hidden_size = hidden_size
    self._output_shape = output_shape

  def __call__(self, z):

    # Lanusse et al. architecture
    # from: https://github.com/makagan/SSI_Projects/blob/main/cf_notebooks/2.PartII-GenerativeModels.ipynb

    # Reshape latent variable to an image
    x = hk.Linear(256)(z)
    x = jax.nn.leaky_relu(x)

    x = hk.Linear(3*3*128)(x)
    x = x.reshape([-1,3,3,128])

    x = hk.Conv2DTranspose(128, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)

    x = hk.Conv2DTranspose(64, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)

    x = hk.Conv2DTranspose(32, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)

    x = hk.Conv2DTranspose(16, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)

    x = hk.Conv2DTranspose(8, kernel_shape=3, stride=2)(x)

    x = hk.Conv2D(1, kernel_shape=5)(x)

    x = x[...,0]
    logits = jnp.pad(x, [[0,0],[3,2],[3,2]])  # This step is to pad the image for the 101x101 expected size

    #z = hk.Linear(self._hidden_size)(z)
    #z = jax.nn.relu(z)

    #logits = hk.Linear(onp.prod(self._output_shape))(z)
    #logits = jnp.reshape(logits, (-1, *self._output_shape))

    return logits


class VAEOutput(NamedTuple):
  image: jnp.ndarray
  mean: jnp.ndarray
  stddev: jnp.ndarray
  logits: jnp.ndarray


class VariationalAutoEncoder(hk.Module):
  """Main VAE model class, uses Encoder & Decoder under the hood."""

  def __init__(self, hidden_size=512, latent_size=10,
               output_shape=IMAGE_SHAPE):
    super().__init__()
    self._hidden_size = hidden_size
    self._latent_size = latent_size
    self._output_shape = output_shape

  def __call__(self, x):
    x = x.astype(jnp.float32)
    mean, stddev = Encoder(self._hidden_size, self._latent_size)(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    pred = Decoder(self._hidden_size, self._output_shape)(z)
    dist = tfd.Independent(tfd.Normal(loc=pred, scale=1.0), reinterpreted_batch_ndims=2)
    #image =  dist.sample(seed=hk.next_rng_key())
    image = pred
    
    #p = jax.nn.sigmoid(logits)
    #image = jax.random.bernoulli(hk.next_rng_key(), p)

    return VAEOutput(image, mean, stddev, pred)
  

# they're not really logits anymore coming out of the decoder hmm...
def log_prob(x: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
  # this assumes some kind of sigma? why 1.0 used by Lanusse?
  # 1.0 is definitely not right...what if sqrt for poisson (to start)
  scale_approx = 1e-5#jnp.sqrt(x + 9e-6)
  dist = tfd.Independent(tfd.Normal(loc=pred, scale=scale_approx), reinterpreted_batch_ndims=2)
  return dist.log_prob(x)


def lensing_log_prob(x: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
  lensed_orig, lensed_decoded = herculens_util.batched_apply_lensing(
    original_ims=x,decoded_ims=pred)
  scale_approx = 1e-6#jnp.sqrt(x + 9e-6)
  dist = tfd.Independent(tfd.Normal(loc=lensed_decoded, scale=scale_approx), 
    reinterpreted_batch_ndims=2)
  return dist.log_prob(lensed_orig)

def kl_gaussian(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""Calculate KL divergence between given and standard gaussian distributions.
  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
           = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
           = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution
  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)


# pylint: disable=unnecessary-lambda
model = hk.transform(lambda x: VariationalAutoEncoder()(x))


@jax.jit
def loss_fun(params, rng_key, batch):
  """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs = model.apply(params, rng_key, batch)
  #log_likelihood = -binary_cross_entropy(batch, outputs.logits)
  log_likelihood = log_prob(batch,outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  # ADD A FACTOR TO REDUCE CONTRIBUTION OF KL-DIV (we will deviate from prior...)
  # TODO: HANDLE!!
  elbo = log_likelihood - 0.0001*kl
  return -jnp.mean(elbo)

@jax.jit
def lensing_loss_fun(params, rng_key, batch):
  """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs = model.apply(params, rng_key, batch)
  #log_likelihood = -binary_cross_entropy(batch, outputs.logits)
  log_likelihood = lensing_log_prob(batch,outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  # ADD A FACTOR TO REDUCE CONTRIBUTION OF KL-DIV (we will deviate from prior...)
  # TODO: HANDLE!!
  elbo = log_likelihood - 0.0001*kl
  return -jnp.mean(elbo)

def return_loss_componenets(params, rng_key, batch):
  # Returns: -log_likelihood term, + KL_div term, nelbo term
  """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""
  outputs = model.apply(params, rng_key, batch)
  #log_likelihood = -binary_cross_entropy(batch, outputs.logits)
  log_likelihood = log_prob(batch,outputs.logits)
  kl = kl_gaussian(outputs.mean, jnp.square(outputs.stddev))
  # ADD A FACTOR TO REDUCE CONTRIBUTION OF KL-DIV (we will deviate from prior...)
  # TODO: HANDLE!!
  elbo = log_likelihood - 0.0001*kl
  return jnp.mean(log_likelihood), jnp.mean(-0.0001*kl), -jnp.mean(elbo)
   

def main():

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  #tf.config.experimental.set_visible_devices([], 'GPU')

  # Initialize solver.
  learning_rate = 1e-3
  # 
  if use_lensing_loss:
    solver = OptaxSolver(opt=optax.adam(learning_rate), fun=lensing_loss_fun)
  else:
    solver = OptaxSolver(opt=optax.adam(learning_rate), fun=loss_fun)

  # Set up data iterators.
  #train_ds = load_dataset(tfds.Split.TRAIN, FLAGS.batch_size)
  #test_ds = load_dataset(tfds.Split.TEST, FLAGS.batch_size)

  # Initialize parameters.
  random_seed = 42
  rng_seq = hk.PRNGSequence(random_seed)
  if init_file is None:
    params = model.init(next(rng_seq), onp.zeros((1, *IMAGE_SHAPE)))
    state = solver.init_state(params,next(rng_seq),onp.zeros((1, *IMAGE_SHAPE)))
  else:
    with open(init_file, 'rb') as file:
        params, state = pickle.load(file)
  #state = optimizer.init_state(params)#,next(rng_seq),onp.zeros((1, *IMAGE_SHAPE)))

  # Load in training data
  tf_dataset = generate_tf_dataset(tfrecord_path,batch_size,n_epochs)

  loss_evolution = []
  batch_num = 10*105 + 1
  for batch in tqdm(tf_dataset):
    batch = tf.squeeze(batch)
    #loss, params, opt_state = update(params, next(rng_seq), opt_state, batch.numpy())
    if use_lensing_loss:
       loss, grads = jax.value_and_grad(lensing_loss_fun)(params, next(rng_seq), batch.numpy())
    else:
        loss, grads = jax.value_and_grad(loss_fun)(params, next(rng_seq), batch.numpy())
    #print(params['variational_auto_encoder/decoder/conv2_d_transpose_3']['w'][0,0,0,:])
    #print(grads['variational_auto_encoder/decoder/conv2_d_transpose_3']['w'][0,0,0,:])
    params, state = solver.update(params=params, state=state,
                                  rng_key=next(rng_seq),
                                  batch=batch.numpy())
    #print(params['variational_auto_encoder/decoder/conv2_d_transpose_3']['w'][0,0,0,:])

    #loss = loss_fun(params, next(rng_seq), batch.numpy())
    loss_evolution.append(loss)

    #ll, kl_div, loss = return_loss_componenets(params, rng_key=next(rng_seq),
    #                              batch=batch.numpy())
    #print('ll term: ', ll)
    #print('kl_div term: ', kl_div)
    #print('loss: ', loss)
  # Returns: -log_likelihood term, + KL_div term, nelbo term

    # save params to a file so we can load them in and do stuff
    if batch_num % 105 == 0:
        print('loss: ', loss)
        with open('model_lensing2_epoch%03d.pckl'%(batch_num//105), 'wb') as file:
            pickle.dump([params, state], file)

    if batch_num > 40 and debug:
       break

    batch_num += 1

  with open('model_lensing2_epoch_last.pckl', 'wb') as file:
    pickle.dump([params, state], file)

  np.savetxt(loss_path,np.asarray(loss_evolution),delimiter=',')
  

if __name__ == "__main__":
    main()