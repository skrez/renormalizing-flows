# Copyright (c) 2022 Mathis Gerdes
# Licensed under the MIT license (see LICENSE for details).

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
import numpy as np
import haiku as hk

from functools import partial
from typing import Callable, Optional


@jax.jit
def cyclic_corr(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        d-dimensional array.
    """
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.mean(arr1 * shifted)

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@jax.jit
def cyclic_tensor(arr1: jnp.ndarray, arr2: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x, y] = arr1[y] arr2[y+x]``.

    x and y are d-dimensional (lattice) indices. The shapes of arr1
    and arr2 must match.
    The sum is executed with periodic boundary conditions.

    Args:
        arr1: d-dimensional array.
        arr2: d-dimensional array.

    Returns:
        2*d-dimensional array."""
    chex.assert_equal_shape((arr1, arr2))
    dim = arr1.ndim
    shape = arr1.shape

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, arr1 * shifted

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    _, c = fn(arr2)
    return c


@partial(jax.jit)
def cyclic_corr_mat(arr: jnp.ndarray) -> jnp.ndarray:
    """Compute ``out[x] = 1/N sum_y arr[x,x+y]``.

    x and y are d-dimensional (lattice) indices.
    `arr` is a 2*d dimensional array.
    The sum is executed with periodic boundary conditions.

    This function is related to `cyclic_tensor` and `cyclic_corr`:
        >>> a, b = jnp.ones((2, 12, 12))
        >>> c1 = cyclic_corr(a, b)
        >>> c2 = jnp.mean(cyclic_tensor(a, b), 0)
        >>> jnp.all(c1 == c2).item()
        True
        >>> outer_product = jnp.einsum('ij,kl->ijkl', a, b)
        >>> c3 = cyclic_corr_mat(outer_product)
        >>> jnp.all(c2 == c3).item()
        True

    Args:
        arr: 2*d-dimensional array. x is the index of the first d
            dimensions, y is the index of the last d dimensions.

    Returns:
        d-dimensional array.
    """
    dim = arr.ndim // 2
    shape = arr.shape[:dim]
    assert shape == arr.shape[dim:], 'Invalid outer_product shape.'
    lattice_size = np.prod(shape)
    arr = arr.reshape((lattice_size,) * 2)

    def _compute_shift(shifted, _, axis, child):
        # first compute out value then shift to next shifted configuration
        _, sub_matrix = child(shifted)
        shifted = jnp.roll(shifted, -1, axis)
        return shifted, sub_matrix

    def _scan_component(axis, child, size):
        body = partial(_compute_shift, axis=axis, child=child)
        return lambda init: jax.lax.scan(body, init, None, size)

    def _base(shifted):
        return None, jnp.trace(arr[:, shifted.flatten()])

    fn = _base
    for axis in range(dim - 1, -1, -1):
        fn = _scan_component(axis, fn, shape[axis])

    idx = jnp.arange(lattice_size).reshape(shape)
    _, c = fn(idx)
    return c.reshape(shape) / lattice_size


@jax.jit
def reverse_dkl(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Reverse KL divergence.

    The two likelihood arrays must be evaluated for the same set of samples.
    This function then approximates ``int_x q(x) log(q(x)/p(x)) dx``.

    If the samples ``x`` are distributed according to ``p(x)``, this
    is the reverse KL divergence.
    (Semon: the above seems wrong. The above should be from q(x).)
    (Reverse kl is zero-forcing, and is what the existing normalizing flow literature uses.)
    If the samples were taken from p(x), the returned value is the negative
    forward KL divergence.
    (forwards kl is zero-avoiding.)

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        Scalar representing the estimated reverse KL divergence.
    """
    return jnp.mean(logq - logp)


@jax.jit
def effective_sample_size(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Compute the ESS given log likelihoods.

    The two likelihood arrays must be evaluated for the same set of samples.
    The samples are assumed to be sampled from ``p``, such that ``logp``
    is are the corresponding log-likelihoods.

    Args:
        logp: The log likelihood of p (up to a constant shift).
        logq: The log likelihood of q (up to a constant shift).

    Returns:
        The effective sample size per sample (between 0 and 1).
    """
    logw = logp - logq
    log_ess = 2*jax.nn.logsumexp(logw, axis=0) - jax.nn.logsumexp(2*logw, axis=0)
    ess_per_sample = jnp.exp(log_ess) / len(logw)
    return ess_per_sample


def moving_average(x: jnp.ndarray, window: int = 10):
    """Moving average over 1d array."""
    if len(x) < window:
        return jnp.mean(x, keepdims=True)
    else:
        return jnp.convolve(x, jnp.ones(window), 'valid') / window


@jax.jit
def our_fft(phis):
  """our rfft conventions.
  Here phi should be a square array of size (..., N, N).
  N is assumed to be even.

  The relation between the phi(n) in our paper is that phi(n) in our paper
  corresponds to phi(m) here where n = m - (N/2+1, ..., N/2+1).
  Thus as n ranges over vectors with entries in [-N/2+1, N/2],
  m ranges over vectors with entries in [0, N].

  This returns a matrix of size N, N
  which should be thought of as \tilde{\phi}(p) in our paper
  as p ranges over vectors with values in [0, N/2].
  The relation is that our_fft(phi(m))[i,j] = \tilde{\phi}((i, j))
  with the relation that \tilde{\phi}(p+e_i N)=\tilde{\phi}(p)."""
  #This code is no longer needed.
  #N = phi.shape[0]
  #sumpi = jnp.broadcast_to(jnp.arange(0, N/2+1), (int(N/2+1), int(N/2+1)))
  #sumpi =sumpi + sumpi.transpose()
  #phases = jnp.exp(2*jnp.pi*(1j)*(-1)*(N/2+1)*sumpi/N)

  #The insertion of ifftshift just happens to correctly deal with the shift from m to n.
  #conj and choice of norm deals with signs in fourier trasform convention.
  #Note that this assumes reality of phi, and when we take inverse
  #we will have to take the real part to get a real signal due to floating
  #point errors.
  return jax.lax.conj(jnp.fft.fftn(jnp.fft.ifftshift(phis, axes=[-1,-2]), axes=[-1,-2], norm="forward"))

@jax.jit
def our_ifft(phips):
  """our rfft conventions.
  Here phip should be an array of size (..., N, N)
  This inverts our_rfft.

  This returns a matrix of size (..., N, N)."""
  #N = (phip.shape[0]-1)*2
  #sumpi = jnp.broadcast_to(jnp.arange(0, N/2+1), (int(N/2+1), int(N/2+1)))
  #sumpi =sumpi + sumpi.transpose()
  #phases = jnp.exp(2*jnp.pi*(1j)*(-1)*(N/2+1)*sumpi/N)
  #phases deals with shift betwen n and m
  #conj and choice of norm deals with signs in fourier trasform convention.
  return jax.lax.real(jnp.fft.fftshift(jnp.fft.ifftn(jax.lax.conj(phips), axes=[-1,-2], norm="forward"), axes=[-1,-2]))

#todo: maybe i shouldnt jit this.
@partial(jax.jit, static_argnums=[0,1])
def hatpsquared2d(N, L):
  #if n % 2 != 0:
  #  raise NotImplementedError("Hat p squared implemented in 2D only for even-size lattices")
  # We won't actually do this since we expect this to be jitted.
  Ns = jnp.arange(0, N)
  Ns = Ns.at[int(N/2+1):].set(Ns[int(N/2+1):]-N)
  p0 = jnp.broadcast_to(Ns, (N, N))
  sinsquareds0 = jnp.square((jnp.sin((jnp.pi * p0) /N)))
  sinsquareds1 = jnp.transpose(sinsquareds0)
  hatpsquared = ((2*N/L)**2)* (sinsquareds0 + sinsquareds1)
  return hatpsquared #this at index (i, j) has the value of hatp^2((p0, p1)), where p0 = p0(i, j) = Ns(i) and and p1 = Ns(j).

#@partial(jax.jit, static_argnums=[1,2])
def sample_complex_unit_normal(seed, N, sample_shape):
  """Samples from the complex unit normal distribution
  over DFTs of a real signal, i.e.
  the distribution over phi(p) with phi(p) =conj(phi(-p))
  (so phi(0) and phi(N/2) are forced to be real)
  and such that otherwise real and imaginary parts are iid from N(0,1).

  Assumes that N is even."""
  samples = jax.lax.complex(jax.random.normal(seed, sample_shape+(N, N)), jax.random.normal(seed, sample_shape+(N, N)))
  #goal: samples[i, j] = 1/sqrt(2)(samples[i, j] + conj(samples[-i, -j]))
  #the 1/sqrt(2) factor is to make the final guys have sigma^2=1 rather than sigma^2=2. 
  #this forces samples[0,0] and samples[N/2, N/2] to be real while all other values can be complex. 
  samples_flipped = samples.copy()
  samples_flipped = samples_flipped.at[..., 1:, :].set(samples_flipped[..., :0:-1, :])
  samples_flipped = samples_flipped.at[..., :, 1:].set(samples_flipped[..., :, :0:-1])
  almost_final = 1/jnp.sqrt(2)*(samples + jax.lax.conj(samples_flipped))
  final = final.at[0,0].set(final[0,0]/jnp.sqrt(2))
  final = final.at[N//2, N//2].set(final[N//2, N//2]/jnp.sqrt(2))
  return final

#@partial(jax.jit, static_argnums=, static_argnames=["speedup", "L"])
def sample_from_p_t(seed, phi0s, t, speedup=1.0, L=1.0, p0=1):
  N = phi0s.shape[-1]
  Omega = 1/L**2 #check from paper
  sample_shape = phi0s.shape[:-2]
  phip0s= our_fft(phi0s)
  samples = sample_complex_unit_normal(seed, N, sample_shape)
  hatpsquared = hatpsquared2d(N, L)
  hatpsquared=hatpsquared.at[0,0].set(p0)
  prefactor = jnp.sqrt(Omega*(1-jnp.exp(-2*hatpsquared*t*speedup))/(2*hatpsquared))
  #print(prefactor)
  real_space_signal = our_ifft((prefactor*samples) + jnp.exp(-hatpsquared*t*speedup)*phip0s)
  #means = jnp.mean(real_space_signal, axis=[-1, -2])
  return real_space_signal#-means[..., None, None] + jnp.mean(phi0s, axis=[-1,-2])[:, None, None]

#@partial(jax.jit, static_argnums=[1,2], static_argnames=["speedup", "L"])
def sample_from_prior(seed, sample_shape, N, speedup=1.0, L=1.0, p0=1):
  Omega = 1/L**2 #check from paper
  samples = sample_complex_unit_normal(seed, N, sample_shape)
  hatpsquared = hatpsquared2d(N, L)
  hatpsquared = hatpsquared.at[0,0].set(p0)
  prefactor = jnp.sqrt(Omega/(2*hatpsquared))
  sample = sample_complex_unit_normal(seed, N, sample_shape)*prefactor
  real_space_signal = our_ifft(sample)
  #means = jnp.mean(real_space_signal, axis=[-1, -2])
  return real_space_signal#-means[..., None, None]

def convert_to_frequency_matrix(nonneg_freqs):
  #we assume that nonneg_freqs is a matrix of size (N/2+1,N/2+1)
  #with frequencies (0, ..., N/2) on each row/col. 
  #do not use in jitted code.
  N2p1 = nonneg_freqs.shape[0]
  N=(N2p1-1)*2
  out = np.zeros((N,N))
  for i in range(0, N2p1):
    for j in range(0, N2p1):
        out[i,j] = nonneg_freqs[i,j]
        if i >=1:
            out[-i, j] = nonneg_freqs[i,j]
        if j >= 1:
            out[i, -j] = nonneg_freqs[i,j]
        if i>=1 and j >=1:
            out[-i, -j] = nonneg_freqs[i,j]
  return out

class GeneralizedCarossoPrior:
    def __init__(self, N, hatpsquared, speedup=1.0, Omega=1.0):
        self.N=N
        self.Omega=Omega
        self.speedup=speedup
        #hatpsquared is a matrix of shape (N, N)
        #upper left corner is frequencies [0, N/2]. 
        #hatpsquared[i, j] should be the value of |\hat{p}|^2 at p=(i,j).
        #here recall that p_0, p_1 take values in [-N/2+1, ..., N/2]
        self.hatpsquared=jnp.array(hatpsquared)

    def sample_from_p_t(self, seed, phi0s, t):
        N=self.N
        Omega = self.Omega #check from paper
        sample_shape = phi0s.shape[:-2]
        phip0s= our_fft(phi0s)
        samples = sample_complex_unit_normal(seed, N, sample_shape)
        hatpsquared = self.hatpsquared
        prefactor = jnp.sqrt(Omega*(1-jnp.exp(-2*hatpsquared*t*self.speedup))/(2*hatpsquared))
        real_space_signal = our_ifft((prefactor*samples) + jnp.exp(-hatpsquared*t*self.speedup)*phip0s)
        return real_space_signal

    def sample(self,
               sample_shape: tuple[int, ...] = (),
               seed: chex.PRNGKey = None) -> jnp.ndarray:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        Omega = self.Omega #check from paper
        samples = sample_complex_unit_normal(seed, self.N, sample_shape)
        hatpsquared=self.hatpsquared
        #hatpsquared = hatpsquared2d(N, L)
        #hatpsquared = hatpsquared.at[0,0].set(p0)
        prefactor = jnp.sqrt(Omega/(2*hatpsquared))
        sample = sample_complex_unit_normal(seed, self.N, sample_shape)*prefactor
        real_space_signal = our_ifft(sample)
        return real_space_signal

    def log_prob(self, phis: jnp.ndarray) -> jnp.ndarray:
        hatpsquared = self.hatpsquared
        #this is to regularize the log probability
        #hatpsquared  = hatpsquared.at[0,0].set(self.p0)
        phips = our_fft(phis) 
        norms = jax.lax.real(phips*jax.lax.conj(phips))
        rescaled_norms = -(hatpsquared/self.Omega) *  norms
        uncorrected_sums = jnp.sum(rescaled_norms, axis=(-1,-2))/2
        N2 = int(int(self.N)/2)
        return uncorrected_sums + (rescaled_norms[:, 0, 0]/2) + (rescaled_norms[:, N2, N2]/2)



class CarossoPrior:
    def __init__(self, N, speedup=1.0, L=1.0, p0=1): #may want to annotate t_max
        """Element-wise independent unit normal distribution.

        This class is meant to be compatible with the ones
        in ``tensorflow_probability.distributions``.

        Args:
            shape: Shape of base space.
        """
        #if (len(shape) != 2) or (shape[0] != shape[1]):
        #  raise NotImplementedError("Shape for Carosso prior should be 2D and square!")
        #self.shape = shape
        #self.N = self.shape[0]
        #self.t_max = t_max
        self.N=N
        self.L = L
        self.speedup=speedup
        self.Omega = (1.0/L)**2
        self.p0=p0
        #self.covariance = self.Omega/2*(1/hatpsquared2d(self.N, self.L))

    def sample(self,
               sample_shape: tuple[int, ...] = (),
               seed: chex.PRNGKey = None) -> jnp.ndarray:
        """Generate a random independent unit normal sample.

        Args:
            sample_shape: Batch shape of the sample.
            seed: Random seed.

        Returns:
            An array of shape ``(*sample_shape, *self.shape)``.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return sample_from_prior(seed, sample_shape, self.N, speedup=self.speedup, L=self.L)

    #possibly modify so that the core computation has been jitted.    
    def log_prob(self, phis: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the log likelihood.

        Args:
            value: A sample from the present distribution.
                The last dimensions must match the shape of the base space.

        Returns:
            The log likelihood of the samples.

            unfortunately this only computes the log likelihood up to a constant. That may not be ok!
        """
        #logp = jax.scipy.stats.norm.logpdf(value)
        hatpsquared = hatpsquared2d(self.N, self.L)
        #this is to regularize the log probability
        hatpsquared  = hatpsquared.at[0,0].set(self.p0)
        phips = our_fft(phis) 
        norms = jax.lax.real(phips*jax.lax.conj(phips))
        rescaled_norms = -(hatpsquared/self.Omega) *  norms
        uncorrected_sums = jnp.sum(rescaled_norms, axis=(-1,-2))/2
        N2 = int(int(self.N)/2)
        return uncorrected_sums + (rescaled_norms[:, 0, 0]/2) + (rescaled_norms[:, N2, N2]/2)



class IndependentUnitNormal:
    def __init__(self, shape: tuple[int, ...]):
        """Element-wise independent unit normal distribution.

        This class is meant to be compatible with the ones
        in ``tensorflow_probability.distributions``.

        Args:
            shape: Shape of base space.
        """
        self.shape = shape

    def sample(self,
               sample_shape: tuple[int, ...] = (),
               seed: chex.PRNGKey = None) -> jnp.ndarray:
        """Generate a random independent unit normal sample.

        Args:
            sample_shape: Batch shape of the sample.
            seed: Random seed.

        Returns:
            An array of shape ``(*sample_shape, *self.shape)``.
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return jax.random.normal(seed, sample_shape + self.shape)

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the log likelihood.

        Args:
            value: A sample from the present distribution.
                The last dimensions must match the shape of the base space.

        Returns:
            The log likelihood of the samples.
        """
        logp = jax.scipy.stats.norm.logpdf(value)
        logp = jnp.sum(logp, axis=tuple(range(-1, -len(self.shape)-1, -1)))
        return logp


def mcmc_chain(
        key: chex.PRNGKey,
        params: hk.Params,
        sample: Callable[[hk.Params, chex.PRNGKey, int], jnp.ndarray],
        action: Callable[[jnp.ndarray], jnp.ndarray],
        batch_size: int,
        sample_count: int,
        initial: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None) \
        -> tuple[jnp.ndarray, float, jnp.ndarray]:
    """Run MCMC chain with optional initial sample._

    Args:
        key: Random key.
        params: Parameters of the sample.
        sample: Function (params, key, batch_size) -> sample,
            where sample is an array of field samples with shape
            (batch_size, L_1, ..., L_d).
        action: A function (field samples) -> action giving the action
            for each of the field samples such that log(p) = -action + const.
        batch_size: Number of field configurations to sample as a batch
            each time new samples are needed. These generated samples are
            then used sequentially as proposals.
        sample_count: Total number of accepted samples to generate.
        initial: Optional initial sample. Tuple of a (single) sample and
            corresponding action.
    Returns:
        Tuple of samples, acceptance rate, last sample.
        The samples are an array of shape (sample_count, L_1, ..., L2).
        The last sample is of the same kind as the `initial` parameter.
    """
    def index_batch(i_batch_key):
        # index into batch & increment index
        i, batch, key = i_batch_key
        new = (batch[0][i], batch[1][i], batch[2][i])
        return i + 1, new, batch, key

    def new_batch(i_batch_key):
        # generate new batch & reset index
        i, batch, key = i_batch_key
        k0, key = jax.random.split(key)
        x, logq = sample(params, k0, batch_size)
        logp = -action(x)
        return index_batch((0, (x, logq, logp), key))

    def mcmc(state, _):
        i, accepted, last, batch, key = state
        i, new, batch, key = jax.lax.cond(
            i < batch_size,
            index_batch,
            new_batch,
            (i, batch, key))

        last_x, last_logq, last_logp = last
        new_x, new_logq, new_logp = new
        k, key = jax.random.split(key)
        rand = jax.random.uniform(k, ())
        p_accept = jnp.exp((new_logp - new_logq) - (last_logp - last_logq))
        new, accepted = jax.lax.cond(
            rand < p_accept,
            # accept
            lambda inp: (inp[1], inp[2] + 1),
            # reject
            lambda inp: (inp[0], inp[2]),
            (last, new, accepted))

        return (i, accepted, new, batch, key), new[0]

    # initial state
    k0, key = jax.random.split(key)
    x, logq = sample(params, k0, batch_size)
    if initial is not None:
        x = x.at[0].set(initial[0])
        logq = logq.at[0].set(initial[1])
    logp = -action(x)

    # state: next batch index, accepted, last, batch, random key
    init = (1, 0, (x[0], logq[0], logp[0]), (x, logq, logp), key)

    state, chain = jax.lax.scan(mcmc, init, None, length=sample_count)
    _, accepted, last, *_ = state
    return chain, accepted / sample_count, last[:-1]
