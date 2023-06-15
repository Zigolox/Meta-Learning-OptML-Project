#!/usr/bin/env python3
import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jax.random import split
from functools import partial

from dataloader import load_task

import matplotlib.pyplot as plt


def evaluation(model, n_test_train, n_test, train_key, test_key, inner_step, inner_optim, inner_opt_state, batch_loss, **kwargs):
    # sample N different sine curves
    N = 100
    M = 100

    A_keys, w_keys = vmap(lambda key: split(key, N))(split(train_key))
    eval_keys = split(test_key, N)

    def evaluate_task(A_key, w_key, eval_key):
        # for one sine curve, sample M different training/testing sets
        train_keys = split(eval_key, M)
        (x_train, y_train), (x_test, y_test) = eqx.filter_vmap(load_task, in_axes=(None, None, 0, None, None, None, None))(
            n_test_train, n_test, train_keys, test_key, A_key, w_key, False
        )
        # train with one inner step
        test_models = eqx.filter_vmap(partial(inner_step, **kwargs), in_axes=(None, 0, 0, None, None, None))(
            model, x_train, y_train, batch_loss, inner_optim, inner_opt_state
        )

        # average of mean squared errors:
        return eqx.filter_vmap(batch_loss)(test_models, x_test, y_test).mean()

    mse = vmap(evaluate_task)(A_keys, w_keys, eval_keys)
    return mse.mean(), mse.std()
