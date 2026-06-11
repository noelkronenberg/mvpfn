# MVPFN

Minimum viable Prior-data Fitted Network (PFN).

Based on the original PFN architecture ([Müller et al., 2021](https://doi.org/10.48550/arXiv.2112.10510)).

## Structure

- [mvpfn/](mvpfn/) 
  - [priors/](mvpfn/priors/)
    - [regression.py](mvpfn/priors/regression.py) linear-function prior
    - [gp.py](mvpfn/priors/gp.py) Gaussian-process prior
  - [model.py](mvpfn/model.py) model architecture
  - [bar_distribution.py](mvpfn/bar_distribution.py) Riemann distribution
  - [train.py](mvpfn/train.py) prior-data fitting
  - [\_\_init\_\_.py](mvpfn/__init__.py) API
- [notebooks/](notebooks/) experiments
