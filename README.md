# Description
An AI model that takes your wardrobe and outputs the best outfits from the items

# Requirements
- Python 3.10 or later
- Conda

# Installtion

## Clone the repo
```bash
mkdir ml_model_4
cd ml_model_4
git clone https://github.com/OliveTreeSap/ml_model_4 ml_model_4
```

## Create the environment from the file
```bash
conda env create -f environment.yml
```

## Activate the environment
```bash
conda activate ml_model_4
```

# Usage

## Data structure
```bash
data/
└── test/   # your wardrobe
    └── items/
        ├── top/ # shirts, hoodie, ...
        ├── bottom/ # shorts, leggings, ...
        └── shoes/
```

## Run app.py in the terminal
```bash
streamlit run app.py
```