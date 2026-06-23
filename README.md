# FedPRC: Prototype Region Calibration Guided Federated Domain Generalization 

A comprehensive federated learning research framework with a focus on **domain generalization (DG)** and **cross-domain federated learning**. This codebase implements **40+ federated learning algorithms** from recent literature, covering standard FL, personalized FL, domain generalization FL, knowledge distillation, prompt-based FL, semi-supervised FL, and test-time adaptation.

---

## 项目简介

FedPRC 是一个面向**域泛化（Domain Generalization）**的联邦学习研究框架。

框架支持 **40+ 种联邦学习算法**，覆盖标准 FL、个性化 FL（PFL）、域泛化 FL、知识蒸馏、Prompt-based FL、半监督 FL、测试时自适应（TTA）等多个研究方向。

---

## Features

- **40+ FL algorithms** implemented with a unified training interface
- **Domain generalization** on multi-domain benchmarks (PACS, Office-Home, Office-31, DomainNet, Digits)
- **Multiple FL paradigms**: standard, personalized, prompt-based, knowledge distillation, semi-supervised, test-time adaptation
- **Flexible non-IID data partitioning** via Dirichlet distribution
- **Model zoo**: CNN, ResNet18, ResNet50, ShuffleNet, VGG, DenseNet, MobileNet, EfficientNet, CLIP, DAGMM, GNN
- **Factory pattern** for easy algorithm/model extension
- **Comprehensive logging**: CSV metrics, TensorBoard, experiment config snapshots

---

## Supported Algorithms

### Domain Generalization (核心方向)
| Algorithm | Paper | Code |
|-----------|-------|------|
| **FedPRC (VDDG)** | Prototype-based Representation Calibration | `core/Server/dg/ServerFedPRC.py` |
| FedGA | Federated Domain Generalization with Generalization Adjustment | `ServerFedGA.py` |
| FedIIR | Federated Invariant Inference and Reasoning | `ServerFedIIR.py` |
| FedGM | Federated Gradient Matching | `ServerFedGM.py` |
| FedSAM | Sharpness-Aware Minimization for Federated Learning | `ServerFedSAM.py` |
| FedOMG | Federated Orthogonal Model Generation | `ServerFedOMG.py` |
| FedSR | Federated Sufficient Representations | `ServerFedSR.py` |
| FedLGF | Federated Local and Global Feature alignment | `ServerFedLGF.py` |
| FedTTA | Federated Test-Time Adaptation | `ServerFedTTA.py` |
| FedAlign | Feature Alignment in Federated Learning | `ServerFedAlign.py` |
| CCST | Cross-Client Style Transfer | `ServerCCST.py` |
| StableFDG | Stable Federated Domain Generalization | `ServerStableFDG.py` |
| FedGS | Federated Gradient Surgery | `ServerFedGS.py` |

### Standard Federated Learning
| Algorithm | Code |
|-----------|------|
| FedAvg | `core/Server/ServerFedAvg.py` |
| FedProx | `core/Server/ServerFedProx.py` |
| MOON | `core/Server/ServerMOON.py` |
| FedBN | `core/Server/ServerFedBN.py` |
| FedAWA | `core/Server/ServerFedAWA.py` |
| FedProto | `core/Server/ServerFedProto.py` |
| FedSCE | `core/Server/fl/ServerFedSCE.py` |

### Personalized Federated Learning (PFL)
FedALA, FedAS, FedBABU, FedDYN, FedETF, FedFDA, FedGPFL, FedNH, FedPAC, FedROD, FedUV

Location: `core/Server/pfl/`

### Knowledge Distillation
FedHKD, FedMD

Location: `core/Server/ks/`

### Prompt-based Federated Learning
FedPrompt, FedOTP, PFedPrompt, FedAvgC, FedProxC, FedOMGC

Location: `core/Server/prompt/`

### Other
- **Semi-supervised**: FedSemi (`core/Server/semi/`)
- **Single-domain Gen**: FedSingle (`core/Server/single/`)
- **Test-time Adaptation**: FedATP (`core/Server/tta/`)

---

## Supported Datasets

### Domain Generalization Benchmarks
| Dataset | Domains | Classes |
|---------|---------|---------|
| **PACS** | Photo, Art Painting, Cartoon, Sketch | 7 |
| **Office-31** | Amazon, DSLR, Webcam | 31 |
| **Office-Home** | Art, Clipart, Product, Real World | 65 |
| **DomainNet** | Clipart, Infograph, Painting, Quickdraw, Real, Sketch | 345 |
| **Digits** | MNIST, USPS, SVHN, SYN | 10 |

### Traditional FL Datasets
CIFAR-10, CIFAR-100, EMNIST, SVHN, Stanford Cars, CUBs, Caltech-10

> **Note**: All datasets are expected in **HDF5 (`.h5`)** format with `images` and `labels` datasets. Each domain has its own `train.h5` and `test.h5` stored under `{domain_path}/{dataset}/{domain}/`.

---

## Requirements

- Python 3.8+
- **PyTorch** (`torch`, `torchvision`)
- **numpy**, **scipy**, **scikit-learn**
- **pandas** (CSV logging)
- **h5py** (HDF5 dataset loading)
- **tensorboardX** (TensorBoard logging)
- **tqdm** (progress bars)
- **calmsize** (memory profiling)
- **pynndescent** (optional, for FINCH clustering)
- **PIL / Pillow** (image transforms)
- **clip** (for prompt-based methods)

Install dependencies:

```bash
pip install torch==1.8.0 torchvision==0.9.0 numpy scipy scikit-learn pandas h5py tensorboardX tqdm calmsize pytorch_clip
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd FedPRC

# Install dependencies
pip install -r requirements.txt

# Prepare datasets
# Place HDF5 files under {domain_path}/{dataset}/{domain}/
# Default domain_path: E:\wj4all\data  (modifiable via --domain_path)
```

---

## Usage

### Quick Start

```bash
# Standard federated learning (FedAvg on CIFAR-10)
python main.py --alg FedAvg --dataset cifar10 --num_clients 10 --comm_round 200

# Domain generalization with FedPRC (leave-one-domain-out on Office-31)
python main.py --alg FedPRC --domain True --domain_dataset office31 --leave_domain 1 --meta_num 3

# FedProx with personalized regularization
python main.py --alg FedProx --dataset cifar10 --lam 0.01 --num_clients 10
```

### Domain Generalization Mode

```bash
# Leave-one-domain-out DG on PACS (test on domain 0)
python main.py \
    --alg FedPRC \
    --domain True \
    --domain_dataset pacs \
    --leave_domain 1 \
    --meta_num 3 \
    --model resnet18 \
    --comm_round 200 \
    --local_ep 3 \
    --lr 0.001 \
    --batch_size 32 \
    --coral 0.1 \
    --uv 0.1 \
    --style 0.1 \
    --domain_path /path/to/data
```



### Prompt-based Federated Learning

```bash
# FedOTP with CLIP backbone
python main.py \
    --alg FedOTP \
    --domain True \
    --domain_dataset office31 \
    --clip_name ViT-B/16 \
    --INPUT_SIZE 224 \
    --n_ctx 7 \
    --num_prompt 2
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alg` | FedSCE | Algorithm name (FedAvg, FedProx, FedPRC, FedProto, ...) |
| `--dataset` | office31 | Dataset name |
| `--model` | resnet18 | Model backbone (CNN, resnet18, shufflenet, clip) |
| `--domain` | True | Enable domain generalization mode |
| `--domain_dataset` | office31 | Domain benchmark (pacs, office31, officehome, domainnet, digits) |
| `--domain_path` | `E:\wj4all\data` | Root path for HDF5 data files |
| `--leave_domain` | 0 | Leave-one-domain-out index (0 = domain adaptation, ≥1 = DG) |
| `--meta_num` | 3 | Number of clients per domain |
| `--num_clients` | 10 | Total number of clients |
| `--comm_round` | 200 | Communication rounds (global epochs) |
| `--local_ep` | 3 | Local training epochs per round |
| `--batch_size` | 32 | Mini-batch size |
| `--lr` | 0.001 | Learning rate |
| `--beta` | 0.1 | Dirichlet distribution alpha (non-IID degree) |
| `--sampling_rate` | 1.0 | Fraction of clients selected per round |
| `--seed` | 10 | Random seed |
| `--code_len` | 64 | Feature embedding dimension |
| `--num_classes` | 7 | Number of classes |
| `--lam` | 0.05 | Hyper-parameter for auxiliary loss |
| `--gamma` | 0.05 | Hyper-parameter for additional loss |
| `--project` | da_runs_result | Output directory name |

### FedPRC-specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--coral` | 0.1 | Coral loss weight (second-order feature alignment) |
| `--uv` | 0.1 | Uniformity loss weight (feature dispersion) |
| `--style` | 0.1 | Style transfer loss weight (AdaIN) |
| `--soft` | 0.5 | Soft prediction temperature |
| `--ug` | 1.0 | Unsupervised regularization weight |
| `--noise` | 1.0 | Noisy client ratio |

### PromptFL Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--clip_name` | ViT-B/16 | CLIP model variant |
| `--n_ctx` | 7 | Number of text prompt tokens |
| `--INPUT_SIZE` | 224 | Input image resolution for CLIP |
| `--num_prompt` | 2 | Number of learnable prompts |
| `--avg_prompt` | 1 | Number of prompts to aggregate |

---

## Project Structure

```
FedPRC/
├── main.py                 # Main entry point
├── option.py               # Command-line argument parser
├── general.py              # Path utilities
├── utils.py                # Accuracy, soft_predict, average_weights
├── forward_step.py         # DAGMM anomaly detection loss
├── mem_utils.py            # PyTorch memory profiler
│
├── core/                   # Core FL framework
│   ├── base.py             # Factory: algorithm name → Server class
│   ├── Server/             # Server-side implementations
│   │   ├── ServerBase.py   # Base server (train loop, aggregation, logging)
│   │   ├── dg/             # Domain Generalization servers (13 algorithms)
│   │   ├── fl/             # Standard FL servers
│   │   ├── pfl/            # Personalized FL servers
│   │   ├── prompt/         # Prompt-based FL servers
│   │   ├── ks/             # Knowledge distillation servers
│   │   ├── semi/           # Semi-supervised FL servers
│   │   ├── single/         # Single-domain generalization servers
│   │   └── tta/            # Test-time adaptation servers
│   ├── Client/             # Client-side implementations (mirrors Server/)
│   └── utils/              # FINCH clustering, utilities
│
├── model/                  # Neural network architectures
│   ├── base.py             # Factory: model name → nn.Module
│   ├── models.py           # General models (CNN, ResNet18, ShuffleNet...)
│   ├── domain_model.py     # Domain-specific architectures
│   ├── stable_model.py     # StableFDG model components
│   ├── wjclip.py           # CLIP wrapper for federated DG
│   ├── dagmm.py            # Deep Autoencoding GMM
│   ├── ae.py               # Autoencoder
│   ├── graph_model.py      # Graph neural network
│   └── clip/               # Local CLIP implementation
│
└── datautil/               # Data loading and partitioning
    ├── cls_datasets.py     # Dataset dispatcher (domain vs. traditional)
    ├── domainsampling.py   # Domain-aware data loading (HDF5)
    └── sampling.py         # Non-IID data partitioning (Dirichlet)
```

---

## Adding a New Algorithm

1. **Create server** in `core/Server/<category>/ServerYourAlg.py`
   - Inherit from `ServerBase` and implement `Create_Clints()` and optionally override `train()`
2. **Create client** in `core/Client/<category>/ClientYourAlg.py`
   - Inherit from `ClientBase` and override `update_weights()` with your local training logic
3. **Register** in `core/base.py`:
   - Add import and `elif args.alg == "YourAlg"` branch in `init_server()`

---

## Output

All results are saved under `{project}/{alg}_{beta}_{seed}_{part}_{dataset}_{leave_domain}_{sampling_rate}_{meta_num}_{uv}_{coral}_{style}/{name}/`:

```
├── logs/
│   ├── main.log                    # Detailed training log
│   ├── experiment_arguments.json   # Full argument snapshot
│   └── TensorBoard events
├── <alg>_<seed>_<...>_results.csv  # Per-round metrics (loss, accuracy)
└── saved_models/                   # Model checkpoints (if --save_model)
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{
  title={Prototype Region Calibration Guided Federated Domain Generalization},
  author={Wenjie Yao},
  year={2026}
}
```

---

