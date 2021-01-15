import  warnings
warnings.simplefilter("ignore")
import os 
import pickle
import pandas as pd

def churn_score(score_df):

    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if len(pkl_files) != 1:
    	raise ValueError('should be only one pickle file in the current directory')

    pklfilename = pkl_files[0]
    pFile = open(pklfilename, "rb")
    thisModelFit = pickle.load(pFile)
    pFile.close()

    #remove/replace special characters and white spaces
    score_df.columns = [i.replace(' ','_').replace("'",'').lower() for i in score_df.columns]

    score_df = score_df.drop(["phone","state","area_code","intl_plan","vmail_plan"],axis=1)
    encoded = pd.get_dummies(score_df)
    X_cols = [i for i in encoded.columns if "churn" not in i.lower()]

    predProb = _thisModelFit.predict(X)

    # Retrieve the event probability
    EM_EVENTPROBABILITY = float(_predProb[1])

    #Determine the predicted target category
    EM_CLASSIFICATION = _predProb.idxmax(axis = 1).to_string(index=False)



	

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)









