# lmdeb
environment setup:

`conda install -c conda-forge jupyterlab`

**matplotlib:**
`conda config --add channels conda-forge`
`conda install matplotlib matplotlib-base mpl_sample_data`

**lightkurve:**
`conda install --channel conda-forge lightkurve`

**exoplanet:**
`conda install -c conda-forge exoplanet`

**theano:**
`conda install numpy scipy mkl mkl-service nose parameterized pygpu`
`pip install git+https://github.com/Theano/Theano.git#egg=Theano`
`conda install theano`

**other/might be important:**
`conda install batman-package libpython`


