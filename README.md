# Re-implementation of Meta Learning Methods

The repository contains code for the reproduction of the results, MAML, FOMAML [1], META-SGD [2] and REPTILE [3] for the EPFL course [Optimization for Machine Learning - CS-439](https://github.com/epfml/OptML_course/tree/master).

The autograd engine [JAX](https://github.com/google/jax), the neural network library [equinox](https://github.com/patrick-kidger/equinox), the optimization library [optax](https://github.com/deepmind/optax) and the tensor operation library [einops](https://github.com/arogozhnikov/einops) are used.

## Requirements

To install requirements locally, run the following command:

```setup
pip install -r requirements.txt
```

## Experiments

The implemented techniques were tested on two different tasks. The first tasks entails replicating a random sin-curve with k-examples. The other problems is to learn to classify images from the [omniglot](https://web.mit.edu/jgross/Public/lake_etal_cogsci2011.pdf) dataset. This is done in a 20-way 1-shot learning setting where the model is trained on one image of 20 different characters and then tested on a new image of the same 20 characters.

To run the experiments, and produce the results, the notebooks in each respective folder can be run.

## Results

Our model achieves the following performance on different tasks:

### Sinusodal 5-shot learning

When training 5 random x,y cordinates sampled from a random sin-curve the model was able to achive the following mean square error: 


| Meta Learning Method         | MSE  |
| --------------- |----------- |
| MAML     | 0.72  |
| META-SGD   | 0.46  |
| FOMAML | 0.81 |
| REPTILE | 1.3 |

### Omniglot 20-way 1-shot learning

When training on 20 different characters from the omniglot dataset and testing on a new image of the same 20 characters the model was able to achive the following accuracy:

| Meta Learning Method         | Accuracy  |
| --------------- |----------- |
| MAML     | 78.9%  |
| META-SGD   | 83.7%  |
| FOMAML | 50.8% |
| REPTILE | 35.8% |

## References
[1] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in International conference on machine learning. PMLR, 2017, pp. 1126–1135.

[2] Z. Li, F. Zhou, F. Chen, and H. Li, “Meta-sgd: Learning to learn quickly for few-shot learning,” arXiv preprint arXiv:1707.09835, 2017.

[3] A. Nichol, J. Achiam, and J. Schulman, “On first-order meta-learning algorithms,” arXiv preprint arXiv:1803.02999, 2018.
