import util
import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import clustering


def runExpDataSet1():



    fname = "Data/water_potability.csv"
    data = util.get_data(fname)
    data = data.dropna()
    X =  data.iloc[:,:-1]
    Y = data.iloc[:,-1]

    clustering.runExpWater(X,Y)

def runExpDataSet2():

    fname = "Data/winequality-red.csv"
    data = util.get_data(fname)
    X =  data.iloc[:,:-1]

    conditions = [
        (data['y'] <= 5),
        (data['y'] == 6),
        (data['y'] > 6)
        ]
    
    values = [1,2,3]
    Y = np.select(conditions, values)

    clustering.runExpWine(X,Y)




def main():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn"
        )
    runExpDataSet1()
    runExpDataSet2()


if __name__ == "__main__":
    main()