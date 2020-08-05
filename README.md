# lmdeb
environment setup:

**matplotlib:**
 `python -m pip install -U pip`
`python -m pip install -U matplotlib`

**lightkurve:**
`conda install --channel conda-forge lightkurve`

**theano:**
`conda install numpy scipy mkl <nose> <sphinx> <pydot-ng>`
  **note**: was unable to use <> in zsh at least, so I used instead:
`conda install numpy scipy mkl theano pygpu nose parameterized`

**pymc3:**
`conda install -c conda-forge pymc3`

**exoplanet:**
`conda install -c conda-forge exoplanet`
**or**
`git clone https://github.com/exoplanet-dev/exoplanet.git`
`cd exoplanet`
`python -m pip install -e .`

**other:**
`conda install batman-package libpython mkl-service`


