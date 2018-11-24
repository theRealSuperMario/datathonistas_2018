README in progress, for now please refer to
- info@childgrowthmonitor.org
- [GitHub main project](https://github.com/Welthungerhilfe/ChildGrowthMonitor/)
- [Child Growth Monitor Website](https://childgrowthmonitor.org)

# Child Growth Monitor Machine Learning

## Introduction
This project uses machine learning to identify malnutrition from 3D scans of children under 5 years of age. This [one-minute video](https://www.youtube.com/watch?v=f2doV43jdwg) explains.

## Getting started

As a first step, please make sure that you work with all the notebooks provided in the skel-folder. This will give you a quick and thourough introduction to the current state-of-the-implementation.

### Requirements
Training the models realistically requires using GPU computation. A separate backend project is currently developing the DevOps infrastructure to simplify this.

You will need:
* Python 3
* TensorFlow GPU
* Keras

### Create documentation
Do this:

```
cd docs
make html
```
Then open docs/build/html/index.html


### Installation of cgmcore module.
You can install the utilities as a module:

```
pip install git+https://github.com/Welthungerhilfe/cgm-ml.git
```

### Installation

#### Linux (Ubuntu)

These steps provide an example installation on a local Ubuntu workstation from scratch:
* Install Ubuntu Desktop 18.04.1 LTS
* Install NVIDIA drivers  
*Please note that after rebooting, the secure boot process will prompt you to authorize the driver to use the hardware via a MOK Management screen.*
```
sudo add-apt-repository ppa:graphics-drivers
sudo apt-get update
sudo apt-get install nvidia-390
sudo reboot now
```
* Install [Anaconda with Python 3.6](https://www.anaconda.com/download)
```conda update conda
conda update anaconda
conda update python
conda update --all
conda create --name cgm
source activate cgm
conda install tensorflow-gpu
conda install ipykernel
conda install keras
conda install vtk progressbar2 glob2 pandas
pip install --upgrade pip
pip install git+https://github.com/daavoo/pyntcloud
```

#### macOS

Tensorflow [dropped GPU support on macOS](https://www.tensorflow.org/install/install_mac). Otherwise the installation is similar to the one on Linux above.

* Install [Anaconda with Python 3.6](https://www.anaconda.com/download)
```conda update conda
conda update anaconda
conda update python
conda update --all
conda create --name cgm
source activate cgm
conda install tensorflow
conda install ipykernel
conda install keras
conda install vtk progressbar2 glob2 pandas
pip install --upgrade pip
pip install git+https://github.com/daavoo/pyntcloud
```

##### Known issues on macOS

1. Saving prepared datasets > 2GB fails

   * Error: `OSError: [Errno 22] Invalid argument`
   * Our current datasets are > 2GB
   * We save them as Pickle files as preparation for the training
   * Python bug ticket: https://bugs.python.org/issue24658
   * Workaround 1: Reduce dataset size
   * Workaround 2: Apply https://stackoverflow.com/a/38003910

##### Experimental GPU support on macOS

Install `tensorflow-gpu` 1.1 via pip: https://www.tensorflow.org/versions/r1.1/install/install_mac.

### Dataset access
Data access is provided on as-needed basis following signature of the Welthungerhilfe Data Privacy & Commitment to
Maintain Data Secrecy Agreement. If you need data access (e.g. to train your machine learning models),
please contact [Markus Matiaschek](mailto:mmatiaschek@gmail.com) for details.

### Preparing training data
* Run the script `python create_datasets.py`

### Training the models
* Run the script `python train_nets.py`

*This currently takes around 6 hours to run on a local NVIDIA GTX 1080 Ti.*

### Evaluating the results
TODO


## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Versioning

Our [releases](https://github.com/Welthungerhilfe/cgm-ml/releases) use [semantic versioning](http://semver.org). You can find a chronologically ordered list of notable changes in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details and refer to [NOTICE](NOTICE) for additional licensing notes and use of third-party components.
