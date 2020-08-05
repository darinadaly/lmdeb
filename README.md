# lmdeb
environment setup:

**jupyter lab:**
`conda install -c conda-forge jupyterlab`

**matplotlib:**
`conda config --add channels conda-forge`
`conda install matplotlib matplotlib-base mpl_sample_data`

**lightkurve:**
`conda install --channel conda-forge lightkurve`

**exoplanet:**
`git clone https://github.com/exoplanet-dev/exoplanet.git
cd exoplanet
python -m pip install -e .`

**other:**
`conda install mkl-service`



