# PATE
In this repository we provide code of the paper:
> **PATE: Property, Amenities, Traffic and Emotions Coming Together for Real Estate Price Prediction**

> Yaping Zhao, Ramgopal Ravi, Shuhui Shi, Zhongrui Wang, Edmund Y. Lam, Jichang Zhao

> arxiv link: https://arxiv.org/abs/2209.05471

The **H4M Dataset** is released at: [https://indigopurple.github.io/H4M/index.html](https://indigopurple.github.io/H4M/index.html)

<p align="left">
<img src="corr.png">
</p>

# Usage
0. For pre-requisites, run:
```
conda env create -f environment.yml
conda activate pate
```
1. To reproduce the results and figures in the paper, run:
```
python main.py
```
2. For further research, visit the website of [H4M Dataset](https://indigopurple.github.io/H4M/index.html).

# Citation
Cite our paper if you find it interesting!
```
@misc{zhao2022pate,
      title={PATE: Property, Amenities, Traffic and Emotions Coming Together for Real Estate Price Prediction}, 
      author={Zhao, Yaping and Ravi, Ramgopal and Shi, Shuhui and Wang, Zhongrui and Lam, Edmund Y. and Zhao, Jichang},
      journal={arXiv preprint arXiv:2209.05471},
      year={2022}
}

@article{zhao2022h4m,
  title={H4M: Heterogeneous, Multi-source, Multi-modal, Multi-view and Multi-distributional Dataset for Socioeconomic Analytics in the Case of Beijing},
  author={Zhao, Yaping and Shi, Shuhui and Ravi, Ramgopal and Wang, Zhongrui and Lam, Edmund Y and Zhao, Jichang},
  journal={arXiv preprint arXiv:2208.12542},
  year={2022}
}
```
