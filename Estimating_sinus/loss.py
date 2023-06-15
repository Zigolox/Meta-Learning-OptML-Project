import equinox as eqx
from jax import vmap

# Typing
from jax import Array


def batch_loss(model: eqx.nn.MLP, x_arr: Array, labels: Array) -> Array:
    """
    Computes the loss for a batch of tasks.

    Args:
        model: Equinox model to take the loss of.
        x_arr: Array of inputs.
        labels: Array of labels.

    Returns:
        Array: Loss for the batch of tasks.
    """

    def loss(x: Array, label: Array):
        # -> (model(x)-Asin(x+w))^2
        pred = model(x)
        return (pred - label) ** 2

    # Mean over the batch
    return vmap(loss)(x_arr, labels).mean()


def multi_batch_loss(model: eqx.Module, batch_loss, batch_of_tasks, inner_step, inner_optim, inner_opt_state) -> Array:
    """
    Compute the loss over multiple models produced by inner_step.

    Args:
        model: Equinox model to take the loss of.
        batch_loss: Loss function to use.
        batch_of_tasks: Batch of tasks to compute the loss over.
        inner_step: Inner step function.
        inner_optim: Inner optimizer.
        inner_opt_state: Inner optimizer state.

    Returns:
        Array: Loss for the batch of tasks.
    """

    # get model_i', test_i, test_labels_i for all tasks i:
    batch_train, batch_train_labels, batch_test, batch_test_labels = batch_of_tasks
    models = eqx.filter_vmap(inner_step, in_axes=(None, 0, 0, None, None, None))(
        model, batch_train, batch_train_labels, batch_loss, inner_optim, inner_opt_state
    )
    # sum loss_i(model_i', test_i, test_labels_i)
    return eqx.filter_vmap(batch_loss)(models, batch_test, batch_test_labels).mean()


def multi_batch_loss_fomaml(models, test, test_labels) -> Array:
    """Compute the loss over all models"""
    return eqx.filter_vmap(batch_loss)(models, test, test_labels).sum()
