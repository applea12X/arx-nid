# arx-nid – Data Catalogue

| Short Name | Domain | Version / DOI | Licence | Size | Citation |
|------------|--------|---------------|---------|------|----------|
| HiTar-2024 | Smart-factory IIoT | [techscience.com DOI:10.32604/cmc.2025.060935](https://www.techscience.com/cmc/v82n2/56568) | CC BY 4.0 | 1.6 GB | Dhaouadi et al., 2025 |
| CIC IoV-2024 | Connected-vehicle CAN | [unb.ca](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) | **see CIC licence** | 3.2 GB | CIC, UNB 2024 |
| LSNM-2024 | Multiclass network attacks | DOI: 10.17632/7pzyfvv9jn.1 | CC BY 4.0 | 2.4 GB | Abu Al-Haija et al., ICICS 2024 |
| CTAP 2018-24 | Human-factors/phishing | Kaggle dataset | CC BY 4.0 | 120 MB | DatasetEngineer, Kaggle 2024 |

> **Tip:** verify SHA-256 after each download; record it in `datasets.csv`.

## Dataset Descriptions

### HiTar-2024
Smart factory Industrial IoT (IIoT) network traffic dataset containing normal and anomalous behaviors in manufacturing environments.

### CIC IoV-2024
Internet of Vehicles (IoV) dataset focusing on Controller Area Network (CAN) bus communications in connected vehicle scenarios.

### LSNM-2024
Large-scale multiclass network attack dataset covering various network intrusion and attack patterns.

### CTAP 2018-24
Comprehensive human factors and phishing detection dataset for cybersecurity awareness training and detection systems.

## Usage

1. Download datasets using scripts in `../scripts/`
2. Raw data stored in `raw/` subdirectory
3. Processed data stored in `processed/` subdirectory
4. All datasets are versioned using DVC for reproducibility
