# Soft Actor-Critic for Variable Speed Limit Control

[![license](https://img.shields.io/badge/license-BSD_3--Clause-gold.svg)](https://github.com/ChocolateDave/a2sos/blob/master/LICENSE)

A variable speed limit control algorithm designed with the Soft Actor-Critic reinforcement learning.

This is a final project for the individual research at UC Berkeley. Please see my [final report](docs/final.pdf) for more details about this project.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies locally.

```bash
git clone -b master --depth 1 https://github.com/ChocolateDave/ce299_fall2022.git
cd ce299_fall2022 & pip install -r requirements.txt & pip install -e .
```

## Usage

A script is provide for running our codes on different penetration rate settings.

```bash
mkdir logs/ & bash ce299/scripts/train_sac_multi_pr.sh
```
    
## Citation

If you use this source code, please cite it using bibtex as below.

```bibtex
@misc{Juanwu2022,
  author = {Juanwu Lu},
  title = {Reinforcement Learning for Freeway Variable Speed Limit Control: A Mixed Traffic Flow Case Study},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ChocolateDave/ce299_fall22}},
  commit = {2b675bac077bc695048ce0072f254de25c898050}
}
```

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE)