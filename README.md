modlm: A toolkit for mixture of distributions language models
==============================================================================
by Graham Neubig (neubig@is.naist.jp)

Install
-------

First, in terms of standard libraries, you must have autotools, libtool, and Boost. If
you are on Ubuntu/Debian linux, you can install them below:

    $ sudo apt-get install autotools libtool libboost-all

You must install Eigen and cnn separately. Follow the directions on the
[cnn page](http://github.com/clab/cnn), which also explain about installing Eigen.

Once these two packages are installed, run the following commands, specifying the
correct paths for cnn and Eigen.

    $ autoreconf -i
    $ ./configure --with-cnn=/path/to/cnn --with-eigen=/path/to/eigen
    $ make

In the instructions below, you can see how to use modlm to train and use language
models.
