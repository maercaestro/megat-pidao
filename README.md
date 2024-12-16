
# MEGAT-PIDAO

This repository contains my implementation of the PIDAO optimizer family, inspired by the paper:

> **[PIDAO: A PID-Based Optimizer for Training Deep Learning Models]**  
*https://www.nature.com/articles/s41467-024-54451-3*

The PIDAO optimizer introduces control theory principles into optimization, including proportional, integral, and derivative components, and extends these with enhanced and adaptive RMS variations.

You can also read on my implementaiton in my Medium/Substack page
https://medium.com/@maercaestro/pidao-a-new-way-of-deep-learning-training-optimization-d1e864dbd237

---

## Repository Structure

The repository includes the following files:

1. **`pidao-base.py`**  
   - Implementation of the base PIDAO optimizer.

2. **`pidao-enhance.py`**  
   - Enhanced PIDAO with velocity integration.

3. **`pidao-rms.py`**  
   - Adaptive RMS-based PIDAO optimizer.

4. **`pidao-trial.ipynb`**  
   - Jupyter Notebook demonstrating the application of PIDAO in:
     - Longer RNN sequence tasks.
     - Quadratic loss landscapes.
     - Simple Neural Networks (MNIST handwritten recognition).

---

## Usage

To use the PIDAO optimizers in your project:
1. Clone this repository:
   ```bash
   git clone https://github.com/maercaestro/megat-pidao.git
   ```
2. Import the desired PIDAO optimizer in your script:
   ```python
   from pidao-base import PIDAO
   from pidao-enhance import EnhancedPIDAO
   from pidao-rms import PIDAccOptimizer_SI_AAdRMS
   ```

---

## Examples

Detailed examples and results are available in **`pidao-trial.ipynb`**. The notebook includes:
- Training longer RNN sequences.
- Optimizing models on quadratic loss functions.
- Applying PIDAO to MNIST handwritten digit classification.

---

## Reference

Please refer to the original paper for the theoretical background of PIDAO:  
*([Paper](https://www.nature.com/articles/s41467-024-54451-3))*

