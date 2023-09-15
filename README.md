# Score-PA: Score-based 3D Part Assembly
This is the official implementation of the paper "Score-PA: Score-based 3D Part Assembly", BMVC 2023 (Oral).

Authors: Junfeng Cheng, Mingdong Wu, Ruiyuan Zhang, Guanqi Zhan, Chao Wu and Hao Dong.

- [Paper Link](https://arxiv.org/abs/2309.04220)

## Installation
### Requirements
- Python 3.8.5
- PyTorch 1.9.1 + CUDA 11.1
- PyTorch Geomtric 2.0.3
- PyTorch3D 0.6.1

### Install Chamfer Distance library
```
cd ./cd
python setup.py install
```

## Datasets
We follow [Generative 3D Part Assembly via Dynamic Graph Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf) to use their preprocessed datasets. Here is the [download link](http://download.cs.stanford.edu/orion/genpartass/prepare_data.zip).

## Usage
We have provided the training and testing scripts for the three datasets (chair, table and lamp) in the folder "scripts". You can directly run the scripts to train and test the model. Please strictly use the parameters in the scripts to reproduce the results in the paper.
### Training
```
sh train_{chair / table / lamp}.sh
```
### Testing
```
sh test_{chair / table / lamp}.sh
```
### Pretrained Models
We have include our pretrained models in the "pretrained_models" folder.

## Citation
```
@article{cheng2023score,
  title={Score-PA: Score-based 3D Part Assembly},
  author={Cheng, Junfeng and Wu, Mingdong and Zhang, Ruiyuan and Zhan, Guanqi and Wu, Chao and Dong, Hao},
  journal={BMVC 2023 (Oral)},
  year={2023}
}
```

## Acknowledgement
Our code is inspired by [Score-based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS) and [Generative 3D Part Assembly via Dynamic Graph Learning](https://proceedings.neurips.cc/paper_files/paper/2020/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf). We thank the authors for their great works.
