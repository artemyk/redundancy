# Implementation of <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20I_\cap^\prec" />

For a detailed description, see:
Artemy Kolchinsky, A Novel Approach to the Partial Information Decomposition, *Entropy*, 2022. [link](https://www.mdpi.com/1099-4300/24/3/403)


This code requires *pypman* library to be installed. It also requires the latest (GitHub) version of *dit*, installed via:
* `pip install https://github.com/dit/dit/archive/master.zip`

For basic usage, see `simpledemo.py`. The code expects a joint distribution over sources and target to be passed in, in [`dit`](https://github.com/dit/dit) format.
