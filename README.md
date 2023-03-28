
# LTVision
LTVision is a package for predicting LTV. It contains two modules
* Module 1 helps answer the question do I need a pLTV model for my data?
* Module 2 helps create the pLTV model (developing in the process)


## Requirements
LTVision requires or works with
* python 3.8.5 or newer.


## Quick start

**1. Installing the package**

Clone repo:
```python
git clone https://github.com/facebookincubator/LTVision.git
```

**2. Creating environment**

  * Creating a new virtual environment:
    ```python
    python3 -m venv venv
    ```

  * Activating a new virtual environment.

    for Mac:
    ```python
    source venv/bin/activate
    ```
    for Windows:
    ```python
    activate venv
    ```

  * Setting up requirements:
    ```python
    pip3 install -r requirements.txt
    ```

  * Running jupyter notebook with created environment.

    To run this step, first make sure that `jupyter notebook`, `ipython` and `ipykernel` packages are installed.
    ```python
    ipython kernel install --user --name=venv
    jupyter notebook
    ```

**3. Getting started**

Use `example.ipynb` for getting started.

To run this notebook with new environment, go to Kernel -> Change kernel -> venv .


## Join the LTVision community
* Website:
* Facebook page:
* Mailing list
* irc:

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
LTVision is licensed under the BSD-style license, as found in the LICENSE file.
