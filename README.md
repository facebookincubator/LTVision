<p align="center">
  <img src="./website/static/img/LTVision-logo.png" alt="logo" width="50%"/>
</p>


# What is LTVision?
--------------------

LTVision is an open-source library from Meta, designed to empower businesses to unlock the full potential of predicted customer lifetime value (pLTV) modeling.

Our vision is to lead the industry by building a community of pLTV practitioners that drives innovation and creates value for everyone in the pLTV ecosystem through expertise, education, and thought leadership.

Our first release - Module 1, is now available. It focuses on generating customer insights and estimating the potential pLTV opportunity size, enabling business decision-makers to evaluate the potential ROI of this initiative before investing in a pLTV model and pLTV strategy.

LTVision is the first step on the pLTV journey. To learn more about Meta’s thought leadership on pLTV, please download our [whitepaper](https://github.com/facebookincubator/LTVision/raw/refs/heads/main/Predicting-LTV-Whitepaper.pdf#).

# Getting Started
-------------------

To get started with LTVision, explore our repository and discover how our library can help you unlock the full potential of pLTV modeling.

* Explore our code base to learn more about LTVision's features and capabilities.
* Share your <a href="https://docs.google.com/forms/d/e/1FAIpQLSej8tQdsuwQ71_cLU-k5s2n933_xuQ5a8pIt1jYtlVcMEmDlA/viewform?usp=sharing">feedback</a> on Module 1
* Join our community to contribute to the development of LTVision and stay up-to-date with the latest updates.

## Requirements
LTVision requires
* python 3.8.5 or newer


## Quick start

**1. Installing the package**

Clone repo:
```python
git clone https://github.com/facebookincubator/LTVision.git
```

**2. Creating an environment**

  * Create a new virtual environment:
    ```python
    python3 -m venv venv
    ```

  * Activate the new virtual environment.

    for Mac:
    ```python
    source venv/bin/activate
    ```
    for Windows:
    ```python
    activate venv
    ```

  * Install requirements:
    ```python
    pip3 install -r requirements.txt
    ```

  * Install pre-commit hooks
    ```
    pre-commit install
    ```

  * Run jupyter notebook with created environment.

    To run this step, first make sure that `jupyter notebook`, `ipython` and `ipykernel` packages are installed.
    ```python
    ipython kernel install --user --name=venv
    jupyter notebook
    ```

**3. Getting started**

Use `example.ipynb` to get started.

To run this notebook with new environment, go to Kernel -> Change kernel -> venv.

# Contribute to LTVision
-------------------------

We welcome contributions from everyone! Developers, data scientists, industry experts and beyond. Your inputs will help shape the future of LTVision and drive innovation in the pLTV ecosystem.

* Report issues or bugs to help us improve LTVision.
* Share your ideas and suggestions for new features or improvements.
* Collaborate with our community to develop new modules and capabilities.

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


# License
--------------------
LTVision is licensed under the BSD-style license, as found in the LICENSE file.


# Disclaimer
-----------------------

LTVision is an open-source project from Meta. Organizations and individuals using LTVision are responsible for running the package on their own infrastructure and with their own data. Meta does not collect or receive any user data through the use of LTVision.

We value your feedback and are committed to continuously improving LTVision. However, please note that Meta may not be able to provide support through our official channels for specific troubleshooting or questions regarding the installation or execution of the package.

<p align="center">
  <img src="./website/static/img/oss_logo.png" alt="logo" width="20%"/>
</p>
