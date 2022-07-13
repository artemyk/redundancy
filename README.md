# Implementation of <img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B200%7D%20I_\cap^\prec" />

For a detailed description, see:
* Artemy Kolchinsky, A Novel Approach to the Partial Information Decomposition, *Entropy*, 2022. [link](https://www.mdpi.com/1099-4300/24/3/403)

For basic usage, see `simpledemo.py`. The code expects a joint distribution over sources and target to be passed in [`dit`](https://github.com/dit/dit) format.

## Installation

This code requires the latest versions of *dit* and *pypoman* libraries to be installed, e.g., via
* `pip3 install --upgrade dit pypoman`

Note that on an Apple M1, the following sequence of commands may be necessary to install `pypoman` (assuming MacBrew is working):
```
brew install gmp
brew install suite-sparse
CFLAGS=-I/opt/homebrew/include/ LDFLAGS=-L/opt/homebrew/lib pip3 install --upgrade pypoman
```
