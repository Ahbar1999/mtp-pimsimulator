# PIMSimulator: ReRAM Wear Leveling Simulator

A Python-based simulator for analyzing write endurance and wear leveling techniques in ReRAM-based Processing-In-Memory (PIM) architectures. This tool integrates directly with **PyTorch**, allowing researchers to simulate real-world neural network training workloads and visualize the physical wear on crossbar arrays.

## 📚 References

This simulator implements logic derived from the following publications.

1.  **ODLPIM**
    > "ODLPIM", 2023. [[Paper Link](https://past.date-conference.com/proceedings-archive/2023/DATA/357.pdf)]

2.  **DRCTL**
    > "DRCTL", 2024. [[Paper Link](https://ieeexplore.ieee.org/document/10764631)]

3.  **TIME**
    > "TIME", 2018. [[Paper Link](https://dl.acm.org/doi/10.1145/3195970.3196071)]

## 🚀 Key Features

* **PyTorch Integration:** Uses `torch.nn.modules` hooks to automatically track writes for weights, gradients, and activations during forward and backward passes.
* **Inter-Crossbar Wear Leveling (Global):** Implements TIWL (Thermal-Aware Inter-block Wear Leveling) approaches to remap "hot" logical data blocks to "cold" physical crossbars.
* **Intra-Crossbar Wear Leveling (Local):** Implements Row Swapping techniques (inspired by TIME and DRCTL papers) to mitigate row-level write hotspots within individual crossbars.
* **Visual Analysis:** Generates heatmaps for Crossbar-level and Row-level wear distributions.
* **Detailed Statistics:** Calculates wear imbalance, reduction percentages, and total write endurance metrics.

## 📋 Prerequisites

* Python 3.8+
* NumPy
* PyTorch & Torchvision
* Matplotlib
* Torchsummary

Install dependencies:
```bash
pip install numpy torch torchvision matplotlib torchsummary
