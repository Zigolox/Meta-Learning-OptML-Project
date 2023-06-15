#!/usr/bin/env python3

from typing import Any, Optional, Sequence, Tuple

import equinox as eqx
import jax.numpy as jnp

# Typing
from jax import Array, lax
from jax import numpy as jnp
from jax import random
from jax.random import PRNGKeyArray
from jax.tree_util import tree_map
from jax_tqdm import scan_tqdm
from dataloader import load_batch_of_tasks

import copy

import matplotlib.pyplot as plt

from loss import multi_batch_loss, multi_batch_loss_fomaml


def train_loop(
    model,
    epochs,
    n_train,
    n_test,
    train_keys,
    test_keys,
    meta_batch_size,
    batch_loss,
    outer_step,
    outer_optim,
    outer_opt_state,
    inner_step,
    inner_optim,
    inner_opt_state,
    **kwargs
):
    """
    Train loop for all meta-learning algorithms.

    Args:
        model: The model to train.
        epochs: Number of epochs to train for.
        n_train: Number of training samples per task.
        n_test: Number of test samples per task.
        train_keys: Jax PRNGKeyArray for generating training tasks.
        test_keys: Jax PRNGKeyArray for generating test tasks.
        meta_batch_size: Number of tasks per batch.
        batch_loss: Loss function for a single task.
        outer_step: Outer update meta learning function.
        outer_optim: Outer optimizer.
        outer_opt_state: Outer optimizer state.
        inner_step: Inner update function.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.
        **kwargs: Additional arguments for the inner update function.

     Returns:
        The trained model.
    """

    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    @scan_tqdm(epochs)
    def scan_fun(carry, it):

        dynamic_model, outer_opt_state = carry
        _, train_key, test_key = it

        model = eqx.combine(dynamic_model, static_model)
        train_keys_batch, test_keys_batch = random.split(train_key, meta_batch_size), random.split(test_key, meta_batch_size)

        batch_of_tasks = eqx.filter_vmap(load_batch_of_tasks, in_axes=(None, None, 0, 0))(n_train, n_test, train_keys_batch, test_keys_batch)

        loss, model, outer_opt_state = outer_step(
            model, batch_loss, outer_optim, outer_opt_state, inner_step, batch_of_tasks, inner_optim, inner_opt_state, **kwargs
        )

        return (eqx.filter(model, eqx.is_array), outer_opt_state), loss

    (dynamic_model, outer_opt_state), losses = lax.scan(scan_fun, (dynamic_model, outer_opt_state), (jnp.arange(epochs), train_keys, test_keys))
    model = eqx.combine(dynamic_model, static_model)
    return model


def step(batch_loss, model, train, train_labels, opt_state, optim):
    """
    Single general step of training.

    Args:
        batch_loss: Loss function for a single task.
        model: The model to train.
        train: Training data.
        train_labels: Training labels.
        opt_state: Optimizer state.
        optim: Optimizer.

    Returns:
        The updated model, optimizer state and loss.
    """

    loss, grad = eqx.filter_value_and_grad(batch_loss)(model, train, train_labels)  # Compute loss and gradient
    updates, opt_state = optim.update(grad, opt_state, model)  # Theta = Theta - alpha*grad(loss)
    model = eqx.apply_updates(model, updates)  # Model = Model(Theta_new)
    return model, opt_state, loss


@eqx.filter_jit
def outer_step_METASGD(model_alpha, batch_loss, outer_optim, outer_opt_state, inner_step, batch_of_tasks, inner_optim, inner_opt_state):
    """
    Outer meta learning update function for METASGD.

    Args:
        model_alpha: The model to train and the alpha parameter.
        batch_loss: Loss function for a single task.
        outer_optim: Outer optimizer.
        outer_opt_state: Outer optimizer state.
        inner_step: Inner update function.
        batch_of_tasks: Batch of tasks.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.

    Returns:
        The updated model, optimizer state and loss.
    """

    loss, grads = eqx.filter_value_and_grad(multi_batch_loss)(
        model_alpha, batch_loss, batch_of_tasks, inner_step, inner_optim, inner_opt_state
    )  # Compute loss and gradient, meta step
    updates, outer_opt_state = outer_optim.update(grads, outer_opt_state, model_alpha)
    model_alpha = eqx.apply_updates(model_alpha, updates)

    return loss, model_alpha, outer_opt_state


def inner_step_METASGD(model_alpha, train, train_labels, batch_loss, inner_optim, inner_opt_state):
    """
    Inner update function for METASGD.

    Args:
        model_alpha: The model to train and the alpha parameter.
        train: Training data.
        train_labels: Training labels.
        batch_loss: Loss function for a single task.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.

    Returns:
        The updated model.
    """
    model, alpha = copy.deepcopy(model_alpha[0]), model_alpha[1]

    _, grad = eqx.filter_value_and_grad(batch_loss)(model, train, train_labels)  # Compute loss and gradient
    mod_grad = tree_map(lambda x, y: x * y, grad, alpha)  # replace standard gradient with alpha*gradient (elementwise product)
    updates, _ = inner_optim.update(mod_grad, inner_opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model


@eqx.filter_jit
def outer_step_MAML(model, batch_loss, outer_optim, outer_opt_state, inner_step, batch_of_tasks, inner_optim, inner_opt_state):
    """
    Outer meta learning update function for MAML.

    Args:
        model: The model to train.
        batch_loss: Loss function for a single task.
        outer_optim: Outer optimizer.
        outer_opt_state: Outer optimizer state.
        inner_step: Inner update function.
        batch_of_tasks: Batch of tasks.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.

    Returns:
        The updated model, optimizer state and loss.

    """

    loss, grads = eqx.filter_value_and_grad(multi_batch_loss)(
        model, batch_loss, batch_of_tasks, inner_step, inner_optim, inner_opt_state
    )  # Compute loss and gradient, meta step
    updates, outer_opt_state = outer_optim.update(grads, outer_opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, outer_opt_state


def inner_step_MAML(model, train, train_labels, batch_loss, inner_optim, inner_opt_state):
    """
    Inner update function for MAML.

    Args:
        model: The model to train.
        train: Training data.
        train_labels: Training labels.
        batch_loss: Loss function for a single task.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.

    Returns:
        The updated model.
    """

    model, inner_opt_state, _ = step(batch_loss, model, train, train_labels, inner_opt_state, inner_optim)

    return model


@eqx.filter_jit
def outer_step_FOMAML(model, batch_loss, outer_optim, outer_opt_state, inner_step, batch_of_tasks, inner_optim, inner_opt_state):
    """
    Outer meta learning update function for FOMAML.

    Args:
        model: The model to train.
        batch_loss: Loss function for a single task.
        outer_optim: Outer optimizer.
        outer_opt_state: Outer optimizer state.
        inner_step: Inner update function.
        batch_of_tasks: Batch of tasks.

    Returns:
        The updated model, optimizer state and loss.
    """

    batch_train, batch_train_labels, batch_test, batch_test_labels = batch_of_tasks
    models = eqx.filter_vmap(inner_step, in_axes=(None, 0, 0, None, None, None))(
        model, batch_train, batch_train_labels, batch_loss, inner_optim, inner_opt_state
    )
    loss, grads = eqx.filter_value_and_grad(multi_batch_loss_fomaml)(models, batch_test, batch_test_labels)  # Compute loss and gradient, meta step

    grad = tree_map(lambda x: x.sum(0), grads)
    updates, outer_opt_state = outer_optim.update(grad, outer_opt_state, model)
    model = eqx.apply_updates(model, updates)

    return loss, model, outer_opt_state


@eqx.filter_jit
def inner_step_FOMAML(model, train, train_labels, batch_loss, inner_optim, inner_opt_state):
    """ "
    Inner update function for FOMAML.

    Args:
        model: The model to train.
        train: Training data.
        train_labels: Training labels.
        batch_loss: Loss function for a single task.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.
    """
    model, inner_opt_state, _ = step(batch_loss, model, train, train_labels, inner_opt_state, inner_optim)

    return model


@eqx.filter_jit
def outer_step_REPTILE(model, batch_loss, _, outer_opt_state, inner_step, batch_of_tasks, inner_optim, inner_opt_state, epsilon, n_steps=32):
    """
    Outer meta learning update function for REPTILE.

    Args:
        model: The model to train.
        batch_loss: Loss function for a single task.
        outer_optim: Outer optimizer.
        outer_opt_state: Outer optimizer state.
        inner_step: Inner update function.
        batch_of_tasks: Batch of tasks.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.
        epsilon: Step size for how much to move towards the new model.
        n_steps: Number of inner optimization steps.
    """
    batch_train, batch_train_labels, _, _ = batch_of_tasks

    models = eqx.filter_vmap(inner_step, in_axes=(None, 0, 0, None, None, None, None))(
        model, batch_train, batch_train_labels, batch_loss, inner_optim, inner_opt_state, n_steps
    )

    model_tilde = tree_map(lambda x: x.mean(0), eqx.filter(models, eqx.is_array))

    dyn_model = tree_map((lambda x, y: x + epsilon * (y - x)), eqx.filter(model, eqx.is_array), eqx.filter(model_tilde, eqx.is_array))
    model = eqx.combine(dyn_model, model)

    return None, model, outer_opt_state


@eqx.filter_jit
def inner_step_REPTILE(model, train, train_labels, batch_loss, inner_optim, inner_opt_state, n_steps=32):
    """
    Inner update function for REPTILE.

    Args:
        model: The model to train.
        train: Training data.
        train_labels: Training labels.
        batch_loss: Loss function for a single task.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.
        n_steps: Number of inner optimization steps.

    Returns:
        The updated model.

    """

    dynamic_model, static_model = eqx.partition(model, eqx.is_array)

    def scan_step(carry, _):
        dynamic_model, inner_opt_state = carry
        model, inner_opt_state, loss = step(batch_loss, eqx.combine(dynamic_model, static_model), train, train_labels, inner_opt_state, inner_optim)
        return (eqx.filter(model, eqx.is_array), inner_opt_state), loss

    carry = (dynamic_model, inner_opt_state)
    (dynamic_model, inner_opt_state), _ = lax.scan(scan_step, carry, jnp.arange(n_steps))
    model = eqx.combine(dynamic_model, static_model)

    return model
