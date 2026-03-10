# Generative Ambisonics Upscaling

[![arXiv](https://img.shields.io/badge/arXiv-2510.00180-b31b1b.svg)](https://arxiv.org/abs/2510.00180)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generative framework for upscaling First-Order Ambisonics (FOA) to High-Order Ambisonics (HOA) using conditional diffusion models.

> **DiffAU: Diffusion-Based Ambisonics Upscaling** > *Amit Milstein, Nir Shlezinger, and Boaz Rafaely* > **Ben-Gurion University of the Negev**

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+


### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Amitmils/GenAU.git](https://github.com/Amitmils/GenAU.git)
   cd GenAU
   ```
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
   3. **Training**
    ```
    python train.py --base_dir <your_base_dir>
    ```



### Citations / References

If you use this code or our research in your work, please cite:
```
@article{milstein2025diffau,
  title={DiffAU: Diffusion-Based Ambisonics Upscaling},
  author={Milstein, Amit and Shlezinger, Nir and Rafaely, Boaz},
  journal={arXiv preprint arXiv:2510.00180},
  year={2025}
}
```
