<p align="center">
  <img src="assets/readme-banner.svg" width="100%" alt="Calibrated KNN-Shapley — A clearer view of training-data value." />
</p>

### On the Inflation of KNN-Shapley Value

**Ziao Yang**, Han Yue, Jian Chen, and Hongfu Liu

**[PDF](https://arxiv.org/pdf/2405.17489)** · **[Paper](https://arxiv.org/abs/2405.17489)** · **[Explore code](#explore-the-code)**

---

Study inflation in KNN-Shapley values and calibrated data valuation across image, text, and tabular datasets.

## Explore the code

| Component | Entry point |
| :--- | :--- |
| Valuation experiments | [`sv.py`](sv.py) |
| Experiment configurations | [`run_sv.sh`](run_sv.sh) · [`config.py`](config.py) |
| Dataset preparation | [`prepare_data.py`](prepare_data.py) |
| Utilities | [`utility_func.py`](utility_func.py) · [`helper.py`](helper.py) |

## Getting started

```bash
git clone https://github.com/yangziao56/Inflation_KNN-SV.git
cd Inflation_KNN-SV
pip install -r requirements.txt
```

Review the dataset settings in [`run_sv.sh`](run_sv.sh), then launch the selected experiments:

```bash
bash run_sv.sh
```

The experiment script includes CKNN configurations for CIFAR-10, tabular datasets, AG News, SST-2, and 20 Newsgroups.

---

<details>
<summary><strong>Citation</strong></summary>

```bibtex
@misc{yang2024inflationknn,
  title = {On the Inflation of KNN-Shapley Value},
  author = {Yang, Ziao and Yue, Han and Chen, Jian and Liu, Hongfu},
  year = {2024},
  eprint = {2405.17489},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2405.17489}
}
```

</details>
