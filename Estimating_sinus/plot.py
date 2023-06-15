#!/usr/bin/env python3
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from einops import rearrange

from dataloader import load_batch_of_tasks


def info_meta_test(expl_model, model, n_test_train, n_test, train_key, test_key, inner_step, inner_optim, inner_opt_state, batch_loss, **kwargs):
    """ "
    Helper function to run an experiment where the model is trained on a task and then seen how it compares to
    the preinitialised model on a new task.

    Args:
        expl_model: The model to create the preinitialised results
        model: The model to train on the task
        n_test_train: The number of training points
        n_test: The number of test points
        train_key: The key to use for the training data
        test_key: The key to use for the test data
        inner_step: The inner step function
        inner_optim: The inner optimiser
        inner_opt_state: The inner optimiser state

    Returns:
        mse: The mean squared error of the model on the test data
        pre_init_plt: The preinitialised model predictions
        gt_plt: The ground truth
        kshot_plt: The k-shot predictions
        scatter_points: The training points

    """

    train_test, train_test_labels, test, test_labels = load_batch_of_tasks(n_test_train, n_test, train_key, test_key, False)

    preds_initial = eqx.filter_vmap(expl_model)(test)

    test_sorted_indices = sorted(range(len(test)), key=lambda k: test[k])

    test_sorted = test[jnp.array(test_sorted_indices)]
    preds_initial_sorted = preds_initial[jnp.array(test_sorted_indices)]
    test_labels_sorted = test_labels[jnp.array(test_sorted_indices)]

    test_model = inner_step(model, train_test, train_test_labels, batch_loss, inner_optim, inner_opt_state, **kwargs)

    preds_shot = eqx.filter_vmap(test_model)(test_sorted)

    mse = jnp.mean(jnp.square(preds_shot.ravel() - test_labels_sorted.ravel()))  # mean squared error

    pre_init_plt = (test_sorted, preds_initial_sorted)  # pre initialization
    gt_plt = (test_sorted, test_labels_sorted)  # ground truth
    kshot_plt = (test_sorted, preds_shot)  # K-shot learning
    scatter_points = (train_test, train_test_labels)  # K points

    return mse, pre_init_plt, gt_plt, kshot_plt, scatter_points


def disp_meta_test(pre_init_plts, gt_plt, kshot_plts, scatter_points, n_train, legends=None, title=''):
    """
    Plotter function to plot the results of the info_meta_test function.
    Compares the preinitialised model to the ground truth and the k-shot learning model.

    Args:
        pre_init_plts: The preinitialised model predictions
        gt_plt: The ground truth
        kshot_plts: The k-shot predictions
        scatter_points: The training points
        n_train: The number of training points
        legends: The legends to use for the plot
        title: The title of the plot

    Returns:
        None
    """

    test_sorted, test_labels_sorted = gt_plt
    plt.plot(test_sorted, test_labels_sorted, color='grey', label="Ground Truth", linewidth=4)

    colors = ['b', 'g', 'r', 'c']
    c = 0
    for pre_init_plt in pre_init_plts:
        test_sorted, preds_initial_sorted = pre_init_plt
        if legends is None:
            plt.plot(test_sorted, preds_initial_sorted, color=colors[c], label="Pre initalization", linestyle='dashed', linewidth=1)
        else:
            plt.plot(test_sorted, preds_initial_sorted, color=colors[c], linestyle='dashed', linewidth=1)
        c += 1

    c = 0
    for kshot_plt in kshot_plts:
        test_sorted, preds_shot = kshot_plt
        if legends is None:
            plt.plot(test_sorted, preds_shot, color=colors[c], label=str(n_train) + "-shot learning", linewidth=2)
        else:
            plt.plot(test_sorted, preds_shot, color=colors[c], label=legends[c], linewidth=2)
        c += 1

    train_test, train_test_labels = scatter_points
    plt.scatter(train_test, train_test_labels, color='black', zorder=2)
    plt.legend()
    plt.title(title)
    plt.show()
