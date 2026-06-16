# MVPFN

Minimum viable Prior-data Fitted Network (PFN).

Based on the original PFN architecture ([Müller et al., 2021](https://doi.org/10.48550/arXiv.2112.10510)).

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/cd39864c-f7ee-4a3e-b127-8b780beeaa5a" />

*Replication of Figure 3 from Müller et al. (2021) using MVPFN.*

## Structure

- [mvpfn/](mvpfn/) 
  - [extensions/](mvpfn/extensions/)
    - [dopfn/](mvpfn/extensions/dopfn/) extension for interventional queries
  - [priors/](mvpfn/priors/)
    - [regression.py](mvpfn/priors/regression.py) linear-function prior
    - [gp.py](mvpfn/priors/gp.py) Gaussian-process prior
    - [causal.py](mvpfn/priors/causal.py) causal (SCM) prior
  - [model.py](mvpfn/model.py) model architecture
  - [bar_distribution.py](mvpfn/bar_distribution.py) Riemann distribution
  - [train.py](mvpfn/train.py) prior-data fitting
  - [\_\_init\_\_.py](mvpfn/__init__.py) API
- [notebooks/](notebooks/) experiments
