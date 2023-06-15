from jax import random
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def load_batch_of_tasks(nr_train, nr_test, train_key, test_key, test_random=True):
    """
    Create a batch of tasks for the sine regression problem.

    Args:
        nr_train (int): Number of training points per task.
        nr_test (int): Number of test points per task.
        train_key (jax.random.PRNGKey): Key for generating training data.
        test_key (jax.random.PRNGKey): Key for generating test data.
        test_random (bool, optional): If True, test points are sampled randomly. If False, test points are sampled uniformly. Defaults to True.

    Returns:
        tuple: Tuple containing:
            train (jnp.ndarray): Training points of shape (nr_train, 1).
            train_labels (jnp.ndarray): Training labels of shape (nr_train, 1).
            test (jnp.ndarray): Test points of shape (nr_test, 1).
            test_labels (jnp.ndarray): Test labels of shape (nr_test, 1).
    """
    A_key, w_key = random.split(train_key)
    A = random.uniform(A_key, shape=(1,), minval=0.1, maxval=5.0)
    w = random.uniform(w_key, shape=(1,), minval=0.0, maxval=jnp.pi)

    train = random.uniform(train_key, shape=(nr_train, 1), minval=-5.0, maxval=5.0)
    train_labels = jnp.concatenate(list(map(lambda x: A * jnp.sin(x + w), train)))

    if test_random:
        test = random.uniform(test_key, shape=(nr_test, 1), minval=-5.0, maxval=5.0).sort()
    else:
        test = jnp.linspace(-5, 5, num=nr_test).reshape(nr_test, 1)
    test_labels = jnp.concatenate(list(map(lambda x: A * jnp.sin(x + w), test)))

    return train, train_labels, test, test_labels


def load_task(n_train, n_test, train_key, test_key, A_key, w_key, test_random=True):
    """
    Create a single task for the sine regression problem.

    Args:
        n_train (int): Number of training points.
        n_test (int): Number of test points.
        train_key (jax.random.PRNGKey): Key for generating training data.
        test_key (jax.random.PRNGKey): Key for generating test data.
        A_key (jax.random.PRNGKey): Key for generating amplitude.
        w_key (jax.random.PRNGKey): Key for generating phase.
        test_random (bool, optional): If True, test points are sampled randomly. If False, test points are sampled uniformly. Defaults to True.

    Returns:
        tuple of (train, train_labels), (test, test_labels) where:
            train (jnp.ndarray): Training points of shape (n_train, 1).
            train_labels (jnp.ndarray): Training labels of shape (n_train, 1).

            test (jnp.ndarray): Test points of shape (n_test, 1).
            test_labels (jnp.ndarray): Test labels of shape (n_test, 1).
    """

    A = random.uniform(A_key, shape=(1,), minval=0.1, maxval=5.0)
    w = random.uniform(w_key, shape=(1,), minval=0.0, maxval=jnp.pi)

    train = random.uniform(train_key, shape=(n_train, 1), minval=-5.0, maxval=5.0)
    train_labels = A * jnp.sin(train + w)
    if test_random:
        test = random.uniform(test_key, shape=(n_test, 1), minval=-5.0, maxval=5.0).sort()
    else:
        test = jnp.linspace(-5, 5, num=n_test).reshape(n_test, 1)
    test_labels = A * jnp.sin(test + w)

    return (train, train_labels), (test, test_labels)
