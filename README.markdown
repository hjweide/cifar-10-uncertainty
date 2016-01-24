# Quantifying Uncertainty in Neural Networks

This is the code used in the experiments described in the blog post [Quantifying
Uncertainty in Neural Networks](http://hjweide.github.io/quantifying-uncertainty-in-neural-networks/).

To run the code, first download the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and
[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) data
sets.  Extract them, and put them in a directory _data_ as _data/cifar-10-batches-py_ and _data/cifar-100-python_.

Configure the parameters in _train.py_ (or leave them as the default) and
create a directory _nets_ for the learned weights to be written to.  Run `python inference.py`
to generate a directory creating the misclassifications from CIFAR-100.  Finally,
run `python train.py` to plot the image grids.

Those interested in further reading, should visit:
[What My Deep Model Doesn't Know...](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)
[Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference](http://arxiv.org/abs/1506.02158)
