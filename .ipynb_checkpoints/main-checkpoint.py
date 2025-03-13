import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
diabetes = pd.read_csv('diabetes_dataset.csv')
print(diabetes.columns)
print(diabetes.head())