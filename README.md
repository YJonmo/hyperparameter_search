## Hyperparameter Optimization Using Optuna for MNIST Classification

This Jupyter notebook demonstrates hyperparameter optimization using Optuna to find optimal model architecture and training parameters for a multi-layer perceptron (MLP) on the MNIST digit classification task.

Key hyperparameters being optimized:
- Learning rate
- Hidden layer sizes 
- Network depth

The optimization process maximizes both classification accuracy and F1 score. Visualization of parameter importance and optimization history helps understand which hyperparameters have the strongest impact on model performance.


## Requirements
- Python 3.10+
- Jupyter Notebook/Lab
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - torch

## Project Structor
```bash
project_root/
│
├── assets/  
├── data/
├── src/
│   ├── models.py
│   └── utils.py     
├── main.ipynb  
├── README.md
└── requirements.txt
```

## Usage
1. Create a virtul env:

```bash
python -m venv my_env
source .my_env/bin/activate
```

2. Install the required dependencies:

```bash
pip install -r requirement.txt
```

3. Run the jupiter notebook (main.ipynb) in colab or locally in VSCode. 


### Sample plots at the end of the jupyer notebook run


<table>
  <tr>
    <td>
      <p>Parallel plot</p>
      <img src="./assets/parallel.jpg" alt="Parallel plot" width="380" height="250">
    </td>
    <td>
      <p>Importance plot</p>
      <img src="./assets/importance.jpg" alt="Importance Plot" width="350" height="250">
    </td>
  </tr>
</table>

