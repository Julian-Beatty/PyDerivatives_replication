# PyDerivatives Replication: Zhang et al. (Asymmetric Returns and Higher Moments)

This repository contains the replication materials for:

 *“The Asymmetric Relationship Between Returns and Implied Higher Moments in Oil”* by Zhang et al.

---
![Oil Implied Surfaces](pyderivatives/Images/oil_implied_surface.png)

## Overview

This replication reproduces and extends the main empirical results using option-implied moments derived from crude oil ETFs.

The repository includes a **frozen version of the PyDerivatives 5.0 package**, ensuring that all results remain fully reproducible regardless of future updates to the main package.

All data processing, estimation, and figure generation are fully automated.

---

## Installation and Replication

To reproduce all results, run the following commands:

```bash
# 1. Create a clean environment
conda create -n pyderivatives_replication python=3.11 -y
conda activate pyderivatives_replication

# 2. Clone the repository
git clone https://github.com/Julian-Beatty/PyDerivatives_replication.git
cd PyDerivatives_replication

# 3. Install the package
pip install -e .

# 4. Run the full replication
python replication.py
```

---
## New Features
![Oil Pricing Kernel Surfaces](pyderivatives/Images/oil_pricing_kernel.png)
