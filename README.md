### CS 7641 Project 3: Unsupervised Learning and Dimensionality Reduction

# Github Link:
https://github.com/jles9/CS7641_p3.git

## Overview
This project seeks to conduct a suvery of unsuprivised learning and dimensionality reduction algorithims, including:
    Clustering:
        k-means clustering
        Expectation Maximization

    Dimensionality Reduction:
        PCA
        ICA
        Randomized Projections
        Another Feature Selection Algo


For this project, each algorithim is analyzed with respect to two different datasets:

    Wine quality: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
    Water potability (https://www.kaggle.com/datasets/adityakadiwal/water-potability)

    Both Datasets are provided under "Data", and are loaded automatically.

## Run instructions
    Ensure sklearn, pandas, np, and matplotlib are installed.  

    Run "runExp.py" from within the project directory.  No flag/config is required

    As it is currently set, runExp.py will automatically run all experiments and generate all graphs and csvs used in the report.

    Experiments for each dataset are broken up into the "runExpDataSet1" (for wine quality) and "runExpDataSet2" (for water quality).  Comment these out in main to disable exp for the dataset
    Experiments for each algorithim are called in a single lines in runExp.py - these may be commented out to avoid performing them



## Files Included
    runExp.py: main run file, runs all experiments and generates reusults

    /results: csv results, labeled per parameter and per algorithm.  "final_" files contain a compact listing of final metrics for an algo/dataset

    /graphs: Contains all automatically generated charts

    /Data: Contains datasets used, along with additional datasets used when implementing algorithims 

    Other included files are incidental and left for later cleanup



## Instructions:
Run runExp.py.  By default, all experiments will be run and will generate a batch of csv values, which may be then used to plot.

python3 ./runExp.py

Each experiment is wrapped by a function, and single experiments may be exlucded from runs by commenting them out in main().

CSV files and figures are saved to folders within the same directory the project is run in.  They may be shown by uncommenting "plt.show()" where it occurs



## Author Information
Student Name: Justin Leszczynski 	  	   		   	 		  		  		    	  
E-mail: jleszczynski9@gatech.edu
GT User ID: jleszczynski9 	   		   	 		  		  		    	 		 	  
GT ID: 903742149 


## Acknowledgements
The Project Specifications the following code was written for are provided by:
Georgia Institute of Technology's CS 7641 (Machine Learning) Class.

Professors: Charles Isbell and Michael Littman

Due Date: November 7th, 2022

Semester: Fall 2022


