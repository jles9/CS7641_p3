import os  		
import unittest
import pandas as pd
import pdb



def get_learner_data_file(basefilename):  		  	   		   	 		  		  		    	 		 		   		 		  
    return open(  		  	   		   	 		  		  		    	 		 		   		 		  
        os.path.join(  		  	   		   	 		  		  		    	 		 		   		 		  
            os.environ.get("LEARNER_DATA_DIR", "Data/"), basefilename  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        "r",  		  	   		   	 		  		  		    	 		 		   		 		  
    )  

def get_data(filename):
    data = pd.read_csv(filename)
    return data

