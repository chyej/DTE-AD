# DTE-AD: Dual-domain Transformer Encoder for Multivariate Time-Series Anomaly Detection

Official PyTorch implementation of the paper:
> "Dual-domain Transformer Encoder with Co-Attention for Multivariate Time-Series Anomaly Detection"

This repository provides the full implementation, datasets preprocessing scripts, and evaluation methods used in our experiments.



##<Environment Setup>
```bash
# Python environment
python >= 3.8
torch >= 1.10
numpy, pandas, scikit-learn, matplotlib
# Install all dependencies
pip install -r requirements.txt




##<Repository Structure>
DTE-AD/DTE-AD
│── main.py                 # Entry point for training and evaluation
│── solver.py               # Model training and evaluation pipeline
│── parameter_num.py        # Parameter counting and model complexity analysis
│
├─data_factory/             # Data loading and preprocessing modules
│   ├── data_loader.py      # Defines dataset segmentation and loaders
│   ├── pklnpy.py           # Converts pickled data to numpy arrays
│
├─model/                    # Model architecture and components
│   ├── Fransformer.py      # Main DTE-AD model definition
│   ├── attn.py             # Attention and co-attention mechanisms
│   ├── embed.py            # Input embedding layers
│   ├── decomposition.py    # Time-series decomposition into trend/residual
│
├─utils/                    # Utility functions
│   ├── utils.py            # Common helper functions
│   ├── pot.py, spot.py     # Threshold estimation (Peak-Over-Threshold)
│   ├── diagnosis.py        # Anomaly diagnosis metrics (HitRate, NDCG)
│   ├── plot.py, tsne.py    # Visualization tools
│
├─scripts/                  # Shell scripts for running all experiments
│   ├── SMD.sh, SWaT.sh, MSL.sh, etc.
│   └── all.sh              # Execute all experiments sequentially
│
└─.idea/                    # IDE configuration (can be ignored)



<Train and Evaluate>
# Example: Run on SWaT dataset
bash scripts/SWaT.sh
# Example: Run on ALL dataset
bash scripts/all.sh
# or run manually
python main.py --anormly_ratio 0.09 --num_epochs 50   --batch_size 256     --mode test    --dataset SWaT   --data_path dataset/SWaT   --input_c 51   --output_c 51   --pretrained_model 20



<Dataset>
SMD, SWaT, PSM, MSL, SMAP, MSDS
Preprocessed data should be placed in ./data/ following each dataset’s folder name.
Sliding windows of size 100 were used, with a stride of 1 for training/validation and non-overlapping windows for testing.

SMD -> https://github.com/NetManAIOps/OmniAnomaly
SWaT -> https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat
PSM -> https://github.com/eBay/RANSynCoders/tree/main/data
MSL -> DTE-AD/DTE-AD/dataset
SMAP -> DTE-AD/DTE-AD/dataset
MSDS -> 
