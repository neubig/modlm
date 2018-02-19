modlm: A toolkit for mixture of distributions language models
==============================================================================
by Graham Neubig (neubig@is.naist.jp)

Docker
------

The dockerfile sets up modlm for training, all dependencies included:
```
docker build -t modlm:latest .
```

Install
-------

First, in terms of standard libraries, you must have autotools, libtool, and Boost. If
you are on Ubuntu/Debian linux, you can install them below:

    $ sudo apt-get install autotools libtool libboost-all

You must install Eigen and dynet separately. Follow the directions on the
[dynet page](http://github.com/clab/dynet), which also explain about installing Eigen.
Note that you should use the version of dynet tagged `v2.0` (commit [1241cfc](https://github.com/clab/dynet/commit/1241cfc3e4e1915d6b7b3914b36b997a8a522397)),
and `eigen` from changeset `346ecdb`, according to the [docs](https://github.com/clab/dynet/blob/1241cfc3e4e1915d6b7b3914b36b997a8a522397/doc/source/install.rst).

```
git clone https://github.com/clab/dynet
cd dynet ; git checkout tags/v2.0 ; cd ..
hg clone https://bitbucket.org/eigen/eigen/ -r 346ecdb

# NOTE Compile dynet, before proceeding with modlm
```

Once these two packages are installed, run the following commands, specifying the
correct paths for dynet and Eigen.

    $ autoreconf -i
    $ ./configure --with-dynet=/path/to/dynet --with-eigen=/path/to/eigen
    $ make

In the instructions below, you can see how to use modlm to train and use language
models.

Citation
--------

More information about the method used in the toolkit can be found in the following paper:

[Generalizing and Hybridizing Count-based and Neural Language Models](http://arxiv.org/abs/1606.00499)
Graham Neubig and Chris Dyer.
ArXiv Preprint.

Training
--------

You can find an example of how to run the toolkit in the `example` directory, which will reproduce our
main experiments from the paper.
Our main experiments can be run by the following process:

* Enter the directory with `cd example`.
* Decompress the training data with `bunzip2 data-ptb/*.bz2`
* Run `preproc.sh` to train count-based language models
* Run `process.sh` to train neurally interpolated n-gram, standard LSTM language model, and neural/ngram hybrid models

Log files and models will be written out to the `result-ptb`

Further instructions about how to use the program are currently in preparation.
