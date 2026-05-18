# Shape Prior Guided Aspect-Ratio Learning for Robust Visual Fall Detection

## Overview

This work introduces a Shape Prior Assisted Learning (SPAL) framework that enhances multi‑scale feature representations. Within this framework, shape prior modules fuse spatial attention and convolution to refine pose features, and an auxiliary supervision signal explicitly predicts the width‑height ratio of each human bounding box, capturing key pose differences between standing and falling states.

##  Dataset

We evaluate our method on public fall detection dataset: https://github.com/zhenghaixia2017/Fall-Detection-Dataset

## Training

```
python train_fall_detection.py
```

## Testing

```
python val.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{shapePrior2026Jmj,
  title={Shape Prior Guided Aspect-Ratio Learning for Robust Visual Fall Detection},
  author={Haoxiang Dong, Mingjie Jiang, Yuxun Wu, Nanxi Li, Haixia Zheng, Mingxia Yang},
  journal={ },
  year={2026}
}
```

## Acknowledgements

We thank the authors of related datasets and open-source projects that made this work possible.

## Contact

For questions or collaborations, please contact: jiangmingjie@qzu.edu.cn














