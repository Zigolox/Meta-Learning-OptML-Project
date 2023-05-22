import jax.numpy as jnp

from distrax import Normal, Bernoulli
from jax.random import split
from jax import vmap
from jax.scipy.special import logsumexp
from einops import reduce, repeat

# Typing
from jax.random import PRNGKeyArray
from jax import Array


def batch_loss(model, x: Array, K: int, key: PRNGKeyArray) -> Array:
    '''Compute the VAE loss.'''

    def loss(x: Array, key: PRNGKeyArray):

        x_rec = model(x, K, key=key)
        log_p_x_z = reduce(Bernoulli(logits=x_rec).log_prob(x), 'c h w -> ', 'sum')

        log_iw = vmap(log_importance_weight)(x_rec, z)
        # Marginalize log likelihood
        return log_iw.mean()

    keys = split(key, x.shape[0])
    # Mean over the batch
    return -vmap(loss_fn)(x, keys).mean()



def vae_loss(model, x: Array, K: int, key: PRNGKeyArray) -> Array:
    '''Compute the VAE loss.'''

    def loss_fn(x: Array, key: PRNGKeyArray):

        x_rec, z, mean, logvar = model(x, K, key=key)

        def log_importance_weight(x_rec, z):
            # Compute importance weights
            log_q_z_x = reduce(Normal(mean, jnp.exp(1 / 2 * logvar)).log_prob(z), 'l -> ', 'sum')
            log_p_z = reduce(Normal(jnp.zeros_like(mean), jnp.ones_like(logvar)).log_prob(z), 'l -> ', 'sum')
            log_p_x_z = reduce(Bernoulli(logits=x_rec).log_prob(x), 'c h w -> ', 'sum')
            return log_p_x_z + log_p_z - log_q_z_x

        log_iw = vmap(log_importance_weight)(x_rec, z)
        # Marginalize log likelihood
        return log_iw.mean()

    keys = split(key, x.shape[0])
    # Mean over the batch
    return -jnp.mean(vmap(loss_fn)(x, keys))


def iwae_loss(model, x: Array, K: int, key: PRNGKeyArray) -> Array:
    '''Compute the IWELBO loss.'''

    def loss_fn(x: Array, key: PRNGKeyArray):

        x_rec, z, mean, logvar = model(x, K, key=key)

        def log_importance_weight(x_rec, z):
            # Compute importance weights
            log_q_z_x = reduce(Normal(mean, jnp.exp(1 / 2 * logvar)).log_prob(z), 'l -> ', 'sum')
            log_p_z = reduce(Normal(jnp.zeros_like(mean), jnp.ones_like(logvar)).log_prob(z), 'l -> ', 'sum')
            log_p_x_z = reduce(Bernoulli(logits=x_rec).log_prob(x), 'c h w -> ', 'sum')
            return log_p_x_z + log_p_z - log_q_z_x

        log_iw = vmap(log_importance_weight)(x_rec, z)
        # Marginalize log likelihood
        return reduce(log_iw, 'k -> ', logsumexp) - jnp.log(K)

    keys = split(key, x.shape[0])
    # Mean over the batch
    return -jnp.mean(vmap(loss_fn)(x, keys))


def old_iwae_loss(model, x, K: int, key: PRNGKeyArray) -> float:
    '''Compute the IWELBO loss.'''

    def loss_fn(x: Array, key: PRNGKeyArray):

        x_rec, _, mean, logvar = model(x, K, key=key)
        # Posterior p_{\theta}(z|x)
        post = Normal(jnp.zeros_like(mean), jnp.ones_like(logvar))

        # Approximate posterior q_{\phi}(z|x)
        latent = Normal(mean, jnp.exp(1 / 2 * logvar))

        # Likelihood p_{\theta}(x|z)
        likelihood = Bernoulli(logits=x_rec)

        # KL divergence
        kl_div = reduce(latent.kl_divergence(post), 'n -> ()', 'sum')

        # Repeat samples for broadcasting
        kl_div = repeat(kl_div, '() -> k', k=K)
        xs = repeat(x, 'c h w -> k c h w', k=K)

        # Log-likelihood or reconstruction loss
        like = reduce(likelihood.log_prob(xs), 'k c h w -> k', 'sum')

        # Importance weights
        iw_loss = reduce(like - kl_div, 'k -> ()', logsumexp) - jnp.log(K)

        return -iw_loss

    keys = split(key, x.shape[0])
    # Mean over the batch
    return jnp.mean(vmap(loss_fn)(x, keys))


def old_vae_loss(model, x: Array, K: int, key: PRNGKeyArray) -> Array:
    '''Compute the VAE loss.'''

    def loss_fn(x: Array, key: PRNGKeyArray):

        x_rec, _, mean, logvar = model(x, K, key=key)

        # Posterior p_{\theta}(z|x)
        post = Normal(jnp.zeros_like(mean), jnp.ones_like(logvar))

        # Approximate posterior q_{\phi}(z|x)
        latent = Normal(mean, jnp.exp(1 / 2 * logvar))

        # Likelihood p_{\theta}(x|z)
        likelihood = Bernoulli(logits=x_rec)

        # KL divergence
        kl_div = reduce(latent.kl_divergence(post), 'n -> ()', 'sum')

        # Log-likelihood or reconstruction loss
        like = reduce(likelihood.log_prob(x), 'k c h w -> k', 'sum')

        # ELBO
        return -(like - kl_div)

    keys = split(key, x.shape[0])
    # Mean over the batch
    return jnp.mean(vmap(loss_fn)(x, keys))
