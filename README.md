# Early_Sepsis_Prediction

## Installation

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

Or if you're using pip3:

```bash
pip3 install -r requirements.txt
```

## Dependencies

The project requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- scipy
- jupyter

All dependencies are listed in `requirements.txt`.

## Troubleshooting

### XGBoost OpenMP Error on macOS

If you encounter an error like:
```
Library not loaded: @rpath/libomp.dylib
```

This means XGBoost needs the OpenMP library. On macOS, you need to install it via Homebrew:

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   Follow the on-screen instructions. After installation, you may need to add Homebrew to your PATH:
   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

2. **Install libomp**:
   ```bash
   brew install libomp
   ```

3. **Reinstall xgboost**:
   ```bash
   pip3 uninstall -y xgboost
   pip3 install xgboost
   ```

### Alternative: Skip XGBoost (if not needed)

If you don't need XGBoost, you can comment out the XGBoost-related code in the notebook. The notebook will work with other models (Random Forest, Logistic Regression, etc.).

## Note

This notebook was originally designed for Google Colab. If running locally:
- Replace Google Colab Drive mount code with local file paths
- Ensure your dataset is accessible at the specified path