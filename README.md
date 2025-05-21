# Exploring NAS-BNN for CIFAR-10 Classification

This repository documents an exploration of the "NAS-BNN: Neural Architecture Search for Binary Neural Networks" framework, applying it to the CIFAR-10 image classification dataset. It includes adapted scripts, custom data preparation utilities, and orchestrator scripts (PowerShell, Python, Jupyter Notebook) to run the complete pipeline.

## Original Work
This work is based on the official NAS-BNN implementation:
- **Paper:** [NAS-BNN: Neural Architecture Search for Binary Neural Networks](https://arxiv.org/abs/2408.15484) (Pattern Recognition 2025)
- **Original GitHub Repository:** [https://github.com/VDIGPKU/NAS-BNN](https://github.com/VDIGPKU/NAS-BNN)
- The original README from the authors can be found in `README-Authors.md`.

## Modifications and Additions
- Adapted model configuration (`superbnn_cifar10`) for CIFAR-10 (32x32 images, 10 classes).
- Created `prepare_cifar10.py` for CIFAR-10 dataset structuring.
- Implemented resume logic and enhanced logging in `search.py`.
- Developed orchestrator scripts to run the full pipeline:
    - `run_all.ps1` (PowerShell for Windows)
    - `run_all.ipynb` (Jupyter Notebook for interactive execution and analysis)
- Diagnostic scripts: `check_ops.py`, `check_cuda.py`, `gpu_load_test.py`.
- Addressed Windows-specific execution issues (e.g., DataLoader workers).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SepehrMohammady/NAS-BNN-CIFAR10-Exploration.git
    cd NAS-BNN-CIFAR10-Exploration
    ```

2.  **Create a Python Environment:**
    It's recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    python -m venv nasbnn 
    # Activate it:
    # Windows (PowerShell): .\nasbnn\Scripts\Activate.ps1
    # Windows (cmd): .\nasbnn\Scripts\activate.bat
    # Linux/macOS: source nasbnn/bin/activate
    ```

3.  **Install PyTorch:**
    Install PyTorch, torchvision, and torchaudio matching your CUDA version. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. For example, for CUDA 12.8 (which `cu128` often implies compatibility with):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    ```
    *(Ensure the versions installed match those in `requirements.txt` after this step).*

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Full Pipeline

This project includes orchestrator scripts to run the entire pipeline. **Ensure your Python environment is activated before running.**

All scripts assume they are run from the root of this repository. Data will be downloaded/prepared into `./data/CIFAR10/` and results will be saved into `./work_dirs/`.

### Option 1: Using PowerShell Script (Recommended for full unattended run)
```powershell
# Ensure your 'nasbnn' (or equivalent) environment is active in this PowerShell session
.\run_all.ps1
```
The script will guide you through steps, including pausing for parameter review.

### Option 2: Using Jupyter Notebook (Good for interactive, step-by-step execution)
1.  Launch JupyterLab or Jupyter Notebook:
    ```bash
    jupyter lab  # or jupyter notebook
    ```
2.  Open `run_all.ipynb`.
3.  Ensure the kernel is set to your `nasbnn` environment.
4.  Run cells sequentially, following the instructions in the markdown cells.


## Pipeline Steps (Automated by Orchestrators)
1.  **Data Preparation (`prepare_cifar10.py`):** Downloads and structures CIFAR-10.
2.  **OPs Range Check (`check_ops.py`):** Determines min/max OPs for `superbnn_cifar10`.
3.  **Supernet Training (`train.py`):** Trains the supernet.
4.  **Architecture Search (`search.py`):** Performs evolutionary search.
5.  **Testing (`test.py`):** Evaluates selected architectures from Pareto front.
6.  **Fine-tuning (`train_single.py`):** Fine-tunes selected architectures.

## Expected Results
After a full run, you can expect:
- A trained supernet checkpoint in `./work_dirs/.../checkpoint.pth.tar`.
- Search results, including the Pareto front, in `./work_dirs/.../search/info.pth.tar`.
- Test logs and fine-tuned model checkpoints in respective subdirectories.
- For the CIFAR-10 setup used, fine-tuned accuracies were in the range of 55-58% Top-1 on the validation set.

## License
The original NAS-BNN code is available under its own license (see `README-Authors.md`). Modifications and scripts added in this repository are provided under the [LICENSE](LICENSE).
