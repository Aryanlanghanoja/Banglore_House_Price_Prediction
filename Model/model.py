# Importing the Necessary Libraries
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

# Function to predicct the prices for the house with the given specification using the trained model
def Predict_Prices(locations , Square_feet , bathroom , BHK) :
    loc_index = np.where(X.columns == locations)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = BHK
    x[1] = Square_feet
    x[2] = bathroom
    
    if loc_index >= 0 :
        x[loc_index] = 1
    
    return model.predict([x])[0]

# FUnction to Convert the Unnecesar format of total_sqft inot usefull format
def clean_totalsqft(x):
    x = str(x)
    if '-' in x:
        parts = x.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    else:
        try:
            return float(x)
        except ValueError:
                return None
            
# Removing the Extereme cases using Standard-deviation
def Clean_Price_Per_Sq_Ft(dataset) :
    dataset_out = pd.DataFrame()
    
    for key,subdf in dataset.groupby('location') :
        mean = np.mean(subdf.Price_Per_Sq_Ft)
        std = np.std(subdf.Price_Per_Sq_Ft)
        reduced_df = subdf[(subdf.Price_Per_Sq_Ft > (mean - std)) & (subdf.Price_Per_Sq_Ft < (mean + std))]
        dataset_out = pd.concat([dataset_out,reduced_df],ignore_index=True)
        
    return dataset_out

# Creating the FUnction to Remove the Data point Having Same Area but more expensive then 3 BHK Appartment
def Remove_BHK_Outlier(dataset) :
    exclude_indices = np.array([])
    
    for location , location_dataframe in dataset.groupby('location') :
        BHK_States = {}
        
        for BHK , BHK_dataframe in location_dataframe.groupby('size') :
            BHK_States[BHK] =  {
                'Mean' : np.mean(BHK_dataframe.Price_Per_Sq_Ft) ,
                'Standard Deviation' : np.std(BHK_dataframe.Price_Per_Sq_Ft) ,
                'Count' : BHK_dataframe.shape[0]
            }
            
        for BHK , BHK_dataframe in location_dataframe.groupby('size') :
            
            statistics = BHK_States.get(BHK - 1)
            
            if statistics and statistics['Count'] > 5 :
                exclude_indices = np.append(exclude_indices , BHK_dataframe[BHK_dataframe['Price_Per_Sq_Ft'] < (statistics['Mean'])].index.values)
                
    return dataset.drop(exclude_indices , axis = 'index')

# Importing the dataset
dataset = pd.read_csv("../Dataset/bengaluru_house_prices.csv")

# Dropping the Unnecessary Features
dataset = dataset.drop(['area_type','society','balcony','availability'],axis='columns')

# Dropping the Columns with NULL Values
dataset = dataset.dropna()

# Changing the location into string from object
dataset['location'] = dataset['location'].apply(lambda x: str(x))

# Removing any leading or following tspace from location
dataset['location'] = dataset['location'].apply(lambda x : x.strip())

# Changing the Format of Features as per required format
dataset['size'] = dataset['size'].apply(lambda x: str(x))
dataset['size'] = dataset['size'].apply(lambda x: int(x.split(' ')[0]))
            
# Apply the cleaning function and Made a total_sqft feature usefull
dataset['total_sqft'] = dataset['total_sqft'].apply(clean_totalsqft)

# Dropping the Columns with NULL Values
dataset = dataset.dropna()

# Implementing Dummy Variables for ENcoding Multiclass Categorical Value
locations = dataset.groupby('location')['location'].count().sort_values(ascending=False)

# Extracting the Location Having Less than 10 occurance
other_locations = locations[locations <= 10]

# Cluubing all the less ocuring location in same category called 'Other'
dataset['location'] = dataset['location'].apply(lambda x : 'Other' if x in other_locations else x)

# Implementing Feature Enginnering and Make New COlumn Price_Per_Sq_Ft for Outlier Detection
dataset['Price_Per_Sq_Ft'] = dataset['price'] * 100000 / dataset['total_sqft']

# Removing the Tuples Having less than 300 sq.ft area per bedroom because it is the busssiness estimate
dataset = dataset[~((dataset['total_sqft'] / dataset['size']) < 300)]

# Analyzing Price_Per_Sq_Ft column for outlier removal
dataset['Price_Per_Sq_Ft'].describe()

# Implementing the Clean_Price_Per_Sq_Ft function
dataset = Clean_Price_Per_Sq_Ft(dataset)

# Calling the Function to Remove the Outliers based on BHK Feature
dataset = Remove_BHK_Outlier(dataset)

# Removing the Houses Having More Number of bathrooms than the bedroom
dataset = dataset[dataset['bath'] < dataset['size'] + 2]

# Drooping the Price per square feet columns because it is for only outlier detection
dataset = dataset.drop('Price_Per_Sq_Ft' , axis='columns')

# Implementing the One Hot ENcoding for encoding the multiclass categorical value
dummies = pd.get_dummies(dataset['location'] , dtype = int)
dataset = pd.concat([dataset , dummies.drop('Other' , axis = 'columns')] , axis = 'columns')
dataset = dataset.drop('location' , axis = 'columns')

# Spliting the dataset into Dependent and Independent Feature
X = dataset.drop('price' , axis = 'columns')
Y = pd.DataFrame(dataset['price'])

# Spliting the dataset into train and test dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 103)

# Creating the Model
model = LinearRegression()

# Training the model
model.fit(X_train,Y_train)

# Evaluting the model
score = model.score(X_test,Y_test)

# Saving the Model into Pickle file and the Structure of test data into json file
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(model,f)

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}

with open("columns.json","w") as f:
    f.write(json.dumps(columns))
    
print("Score :- " , score)
print("Coefficient :- " , model.coef_)
print("Intercept :- " , model.intercept_)
