import pandas as pd
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, make_scorer,auc, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split # Import train_test_split function



churn_df=pd.read_csv('data_churn.csv')
#remove/replace special characters and white spaces
churn_df.columns = [i.replace(' ','_').replace("'",'').lower() for i in churn_df.columns]
#Check out how the resultant dataset look like
churn_df.head()




params = {
          'selectpercentile__percentile': [10,15,20,30,40, 50],
          'randomforestclassifier__max_depth': [5, 7, 9],
          'randomforestclassifier__criterion': ['entropy', 'gini']}



pipe = make_pipeline(
    MinMaxScaler(),
    SelectPercentile(),
    RandomForestClassifier())

grid = GridSearchCV(pipe, scoring= make_scorer(accuracy_score), param_grid=params, cv=5)


# Move CAS Data from a CASTable to a DataFrame

# split the data into train and test set
train_df, valid_df = train_test_split(churn_df, test_size=0.3, random_state=42, shuffle=True)
# train_df = churn_df[churn_df["_PartInd_"]==1].to_frame()
# valid_df = cpart[cpart["_PartInd_"]==0].to_frame()
train_df = train_df.drop(["phone","state","area_code","intl_plan","vmail_plan"],axis=1)
valid_df = valid_df.drop(["phone","state","area_code","intl_plan","vmail_plan"],axis=1)
encoded = pd.get_dummies(train_df)
X_cols = [i for i in encoded.columns if "churn" not in i.lower()]
X = encoded[X_cols].values
y = train_df["churn"]
#Show what happens when cardinalty isnt 


grid.fit(X,y) # <<<-- This is the function we care about


model_combinations_tested = len(grid.cv_results_["params"])
print(f"""Total Tested Combinations {model_combinations_tested} \n
******
Best Params Combo :- {grid.best_params_}
****** 
which yields a best accuracy score {grid.best_score_}""")

model = grid.best_estimator_
model.steps


# Now use the Validation Data (we're using it here like test data) to pick up stats

validation_data_encoded = pd.get_dummies(valid_df)
valid_x_cols = [i for i in validation_data_encoded.columns if "churn" not in i.lower()]
valid_X = validation_data_encoded[valid_x_cols].values
valid_y = valid_df["churn"]
auc_roc = roc_auc_score(pd.to_numeric(valid_y),model.predict_proba(valid_X)[:,1])
print(f"""The accuracy for the model on the validation dataset is {accuracy_score(valid_y,model.predict(valid_X))}""")
print(f"""The Area Under the curve (ROC) on the validation dataset is {auc_roc}""")


# Bring all the classifers together and rank them

sk_model = dict(model = ["GridSearch-Forest-SkLearn"] , C = [auc_roc])
sk_model_df = pd.DataFrame.from_dict(sk_model)
model_rank_df = pd.concat([sk_model_df]).sort_values(by="C",ascending=False)
model_rank_df.index=model_rank_df["C"].rank(method="max",ascending = False)
model_rank_df.index.name = "Rank"
model_rank_df.columns = ["Classifier", "AUC"]
model_rank_df


# Finally, when we need persistence, we'd serialize the model object
# * CAS Models as ASTORE files
# * Python Models as Pickle Files
# 
# Let's do this and perform some batch scoring

from pickle import dump, load
dump(model, open('churn_rf_pipeline_model.pkl','wb'))

print(f"""Pickle file is stored in the working directory""")


