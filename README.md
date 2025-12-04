# Predicting Diabetes Risk from Lifestyle and Demographics
By Hanna Pan

## Installation & Setup
1. **Install Python 3.9+**  
   Recommended via the [Anaconda Distribution](https://www.anaconda.com/products/distribution) (includes Jupyter and most packages).

2. **Set up the CSE 163 environment**  
   Follow instructions from the course site to create the environment and install dependencies:  
   [CSE 163 Software Setup](https://courses.cs.washington.edu/courses/cse163/software/)

3. **Download cse163_utils.py**
    Go to any Take-Home Assessment and download the `cse163_utils.py` file.

3. **Install VSCode (optional)**  
   Download: [https://code.visualstudio.com/](https://code.visualstudio.com/)
 
## Relevant Files
- `eda.py` — Main script: loads data, runs EDA, statistical tests, trains models, and generates outputs
- `utils.py` — Helper functions for data processing, plotting, and ML tasks
- `cse163_utils.py` - File that helps define assert_equals functions for testing
- `tests.py` — Testing file used to test and ensure project code accuracy/validity
- `diabetes.csv` — The dataset this project will study and analyze
- `test_data.csv` - Small testing file to help test project code on a smaller scale
- `report.pdf` — Final written project report

## Set up and How to Run
1. **Create a project folder**
    Create a project folder to be the root of all files and open that folder in a code editor (VScode for example)

2. **Create folders within project folder**
    (Option 1) Create folders inside of the root directory to organize files using this terminal command (NOTE: if you open the root folder in a code editor and access the terminal there, you will already be in the root directory, so there is no need to navigate to the right directory):
```bash
mkdir data notebooks scripts
```

    (Option 2) If you choose not to open the folder in a code editor, navigate to the root folder by typing this command (NOTE: final-project can be substituted by the name of your root folder):
```bash
cd final-project
```

3. **Create relevant files for each folder**
    Add relevant files to each folder, starting with the scripts folder:

```bash
cd scripts
touch eda.py tests.py utils.py
```

Add the already downloaded `cse163_utils.py` to the `scripts` folder.

Add `test_data.csv` to the `data` folder for testing later on. This data will consist of a few lines of data simulating the real dataset to ensure the project works. 

4. **(Optional): create a jupyter notebook**
    Navigate back to the root directory and open a jupyter notebook to help prototype:
```bash
cd ..
jupyter notebook
```
5. **(Optional): open jupyter notebook for prototyping**
    Opening this jupyter notebook will open a browser window. 
    - Go to the `notebooks/` folder
    - Click "New" → "Python 3" to create a new notebook
    - Feel free to rename it to something like eda.ipynb

6. **Download dataset**
    Download the dataset from Kaggle: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data. Unzip it, there will be three csv files to choose from. Choose the one with diabetes_012 in the file name. Feel free to rename it (something like `diabetes.csv` works fine) and move it to the folder called `data`.

7. **(Optional): prototype in jupyter notebook (not require but helpful)**
    Prototype in jupyter notebook to run preliminary EDA and challenge tasks

8. **(Optional): Transfer notebook content to .py scripts**
    If you prototyped on jupyter notebook, transfer jupyter notebook content to .py scripts. `eda.py` will contain all EDA, modeling, and validity functions. `utils.py` will contain any helper methods as needed for `eda.py`

9. **Import relevant libraries and packages**
    Make sure to import all relevant libraries/packages (NOTE: all these packages come with the cse163 environment):
    - `pandas`
    - `seaborn`
    - `matplotlib.pyplot`
    - `numpy`
    - `scipy.stats`
    - `train_test_split` from `sklearn.model_selection`
    - `LogisticRegression` from `sklearn.linear_model`
    - `RandomForestClassifier` from `sklearn.ensemble`
    - `spearmanr` from `scipy.stats`
    - `load_data`, `check_missing`, `data_size`, `data_summary`, `create_diabetes_binary` from `utils`

10. **Select correct python environment to run code**
    Make sure the python scripts are running in the correct environment. Click the search bar above the files, and click "Show and Run Commands >". Find "Python: Select Interpreter" and choose the cse163 environment. Again, set-up instructions can be found here: https://courses.cs.washington.edu/courses/cse163/software/


