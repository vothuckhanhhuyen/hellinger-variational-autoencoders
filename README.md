# Hellinger Multimodal Variational Autoencoders
Official PyTorch implementation for HELVAE, published at **AISTATS 2026**.

This repository is based on the implementation of the ICLR 2021 paper **[Generalized Multimodal ELBO](https://github.com/thomassutter/MoPoE)**.

## Preliminaries

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:
```bash
conda env create -f environment.yml  # create conda env
conda activate mopoe                 # activate conda env
```

Second, download the data, inception network, and pretrained classifiers:
```bash
curl -L -o tmp.zip https://drive.google.com/drive/folders/1lr-laYwjDq3AzalaIe9jN4shpt1wBsYM?usp=sharing
unzip tmp.zip
unzip celeba_data.zip -d data/
```
Please note that all the material downloaded are from the official implementation of the ICLR 2021 paper [Generalized Multimodal ELBO](https://github.com/thomassutter/MoPoE).

## Experiments

To select between the available models (MVAE, MMVAE, MoPoE, HELVAE, and MoHELVAE), set the script variable `METHOD` to one of the following values:
```
"poe" | "moe" | "joint_elbo" | "helvae" | "joint_helvae"
```
By default, each experiment runs with `METHOD="helvae"`.

### Running Bimodal Celeba
```bash
./job_celeba_helvae.sh
```

### Notebook for Toy Dataset Example
```
toy_dataset_helvae.ipynb
```

## Citing 
```
@inproceedings{
vo2026hellinger,
title={Hellinger Multimodal Variational Autoencoders},
author={Huyen Thuc Khanh Vo and Isabel Valera},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=mxHyYltMUa}
}
```

#### Acknowledgements
We thank the authors of the [MoPoE](https://github.com/thomassutter/MoPoE) repo, from which our codebase is based on.
