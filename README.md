# MultIVAP defense against adversarial examples

This repository contains a reference implementation for our defense against [adversarial examples](https://adversarial-ml-tutorial.org/introduction/), which uses a technique from conformal prediction called *inductive Venn-ABERS predictors* (IVAPs; see [1, 2, 3] for more details). This is a follow-up to our ESANN 2019 contribution where we proposed a similar technique that is limited to *binary* classification only [4]. Here, we extend the method to the multiclass case and improve upon its clean accuracy and adversarial robustness.

## Prerequisites

We tested our code using the following dependencies:

* Python 3.6.8 (https://www.python.org/);
* TensorFlow 1.14.0 (https://www.tensorflow.org/);
* Keras 2.2.4 (https://www.keras.io/);
* NumPy 1.16.4 (http://www.numpy.org/);
* Foolbox 1.8.0 (https://foolbox.readthedocs.io);
* CVXOPT 1.2.3 (https://cvxopt.org/documentation/index.html).

There is a requirements file included, so if you have the correct Python version you should be able to install all other dependencies using `pip install -r requirements.txt`.

## Running the code

There are two ways our code can be run: either you follow the Jupyter notebook `cifar10_demo.ipynb` included in this repository or you can use the command-line script. The notebook is self-explanatory; the command-line script can be invoked as follows:

    python main.py cifar10 --batch_size 128 --epochs 10 --frac .8 --eta .03

The first argument is the only mandatory one and must name a Python module, in this case `cifar10.py`. This module must define at least the following methods:

* `get_optimizer`. Returns a Keras optimizer to be used for fitting the model.
* `load_datasets`. Returns a tuple `(x_train, y_train, x_test, y_test)` specifying the training and test data.
* `create_model`. Returns the Keras model that will be trained and used for instantiating the MultIVAP.

The other arguments are optional:

* `batch_size`. Specifices the batch size to use when processing sets of samples in training and inference.
* `epochs`. Number of epochs to train the model.
* `frac`. Maximum fraction of GPU memory to use.
* `eta`. Maximum &ell;<sub>&infin;</sub> perturbation budget for adversarial attacks.

Note that the command-line script will generate a Markdown report under `reports/module.md` where `module` is the module name (in this case, `cifar10`) as well as several plots in the directory `plots/`. You should be able to directly run our code for MNIST, Fashion-MNIST and CIFAR-10 if you have all of the dependencies and if the `plots` and `reports` directories exist. The SVHN and Asirra data sets are not included in Keras and should be downloaded separately:

* [Download link for SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Download link for Asirra](https://www.microsoft.com/en-us/download/details.aspx?id=54765)

The IVAP implementation we use here is provided by [Paolo Toccaceli](https://github.com/ptocca/VennABERS).

## References

1. Vovk, Vladimir, Ivan Petej, and Valentina Fedorova. "Large-scale probabilistic predictors with and without guarantees of validity." Advances in Neural Information Processing Systems. 2015. [PDF](https://papers.nips.cc/paper/5805-large-scale-probabilistic-predictors-with-and-without-guarantees-of-validity.pdf)
2. Vovk, Vladimir, Alex Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer Science & Business Media, 2005.
3. Shafer, Glenn, and Vladimir Vovk. "A tutorial on conformal prediction." Journal of Machine Learning Research 9.Mar (2008): 371-421. [PDF](http://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)
4. Peck, Jonathan. "Detecting adversarial examples with inductive Venn-ABERS predictors." European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning. 2019. [PDF](https://biblio.ugent.be/publication/8622378/file/8622388.pdf)