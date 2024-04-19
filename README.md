<p align="center">

  <h2 align="center">PuzzleAvatar:<br> Assembly of Avatar from Unconstrained Photo Collections</h2>
  <p align="center">
    <a href="https://xiuyuliang.cn/"><strong>Yuliang Xiu</strong></a>
    ·  
    <a href="https://judyye.github.io/"><strong>Yufei Ye</strong></a>
    ·
    <a href="https://itszhen.com/"><strong>Zhen Liu</strong></a>
    ·
    <a href="https://dtzionas.com/"><strong>Dimitris Tzionas</strong></a>
    ·
    <a href="https://ps.is.mpg.de/person/black"><strong>Michael J. Black</strong></a>
    <br>
  </p>
  <h2 align="center">arXiv 2024</h2>
  <!-- <div align="center">
    <video autoplay loop muted src="" type=video/mp4>
    </video>
  </div> -->

  <p align="center">
  </br>
    <a href="">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/PuzzleAvatar-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'></a>
    <a href=""><img alt="youtube views" title="Subscribe to my YouTube channel" src="https://img.shields.io/youtube/views/SjzQ6158Pho?logo=youtube&labelColor=ce4630&style=for-the-badge"/></a>
  </p>
</p>

<br/>


<br/>

## Installation

Please follow the [Installation Instruction](docs/install.md) to setup all the required packages.

## Getting Started

We provide a running script at `scripts/run.sh`. Before getting started, you need to set your own environment variables of `CUDA_HOME`.

After that, you can use TeCH to create a highly detailed clothed human textured mesh from a single image, for example:

```shell
sh scripts/run.sh input/examples/name.img exp/examples/name
```

The results will be saved in the experiment folder `exp/examples/name`, and the textured mesh will be saved as `exp/examples/name/obj/name_texture.obj`

It is noted that in "Step 3", the current version of Dreambooth implementation requires 2\*32G GPU memory. And 1\*32G GPU memory is efficient for other steps. The entire training process for a subject takes ~3 hours on our V100 GPUs.


## Citation

```bibtex
@inproceedings{xiu2024puzzleavatar,
  title={{PuzzleAvatar: Assembly of Avatar from Unconstrained Photo Collections}},
  author={Xiu, Yuliang and Ye, Yufei and Liu, Zhen and Tzionas, Dimitrios and Black, Michael J.},
  booktitle={arXov},
  year={2024}
}
```

## Acknowledgment
This implementation is mainly built based on [TeCH](https://github.com/huangyangyi/TeCH), [BOFT-DreamBooth](https://github.com/huggingface/peft/blob/main/examples/boft_dreambooth/train_dreambooth.py), [Stable Dreamfusion](https://github.com/ashawkey/stable-dreamfusion), [ECON](https://github.com/YuliangXiu/ECON)

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No.860768 ([CLIPE Project](https://www.clipe-itn.eu)).

## Contributors

Kudos to all of our amazing contributors! PuzzleAvatar thrives through open-source. In that spirit, we welcome all kinds of contributions from the community.

<a href="https://github.com/yuliangxiu/PuzzleAvatar/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=yuliangxiu/PuzzleAvatar" />
</a>

_Contributor avatars are randomly shuffled._

---

<br>

## License

This code and model are available for non-commercial scientific research purposes as defined in the [LICENSE](LICENSE) file. By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).

## Disclosure

MJB has received research gift funds from Adobe, Intel, Nvidia, Meta/Facebook, and Amazon. MJB has financial interests in Amazon, Datagen Technologies, and Meshcapade GmbH. While MJB is a part-time employee of Meshcapade, his research was performed solely at, and funded solely by, the Max Planck Society.

## Contact

For technical questions, please contact yuliang.xiu@tue.mpg.de

For commercial licensing, please contact ps-licensing@tue.mpg.de

