# ml-structure-bias
Code for "Training data composition affects performance of protein structure analysis algorithms", published at the Pacific Symposium on Biocomputing 2022

### Software

For protein sequence design and estimation of model accuracy, we use the original Pytorch implementation of Geometric Vector Perceptrons from [Jing et al. (2020)](http://arxiv.org/abs/2009.01411) and [Jing et al. (2021)](http://arxiv.org/abs/2106.03843). To install GVP, clone the repo and pip install: 

    git clone https://github.com/drorlab/gvp-pytorch
    cd gvp-pytorch
    pip install -e .
