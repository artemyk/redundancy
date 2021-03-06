# Implementation of <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20I_\cap^\star" />

For a detailed description, see:
A Kolchinsky, A novel approach to multivariate redundancy and synergy, 2019, [arxiv](https://arxiv.org/abs/1908.08642)

This code requires the Python version of *Parma Polyhedra Library (PPL)* to be installed. This can be done in the following manner:

* `pip install cysignals --user`
* `pip install gmpy2==2.1.0a4 --user`
* `pip install pplpy`

It also requires the latest (GitHub) version of *dit*, installed via:
* `pip install https://github.com/dit/dit/archive/master.zip`

For basic usage, see `simpledemo.py`. The code expects a joint distribution over sources and target to be passed in, in [`dit`](https://github.com/dit/dit) format.
