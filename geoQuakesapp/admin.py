from django.contrib import admin
from datetime import datetime
import pandas as pd
import numpy as np 
from geoQuakesapp.models import Quake, Quake_Predictions
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Register your models here.
admin.site.register(Quake)
admin.site.register(Quake_Predictions)


#check if database table is empty before adding data
if Quake.objects.all().count() == 0:
    df = pd.read_csv(r"C:\Users\bpraf_000\Downloads\database.csv")

    #preview df
    #print(df.head())

    df_load = df.drop(['Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 
    'Root Mean Square', 'Source', 'Location Source', 'Magnitude Source', 'Status'], axis=1)

    #preview df_load
    #print(df_load.head())

    df_load = df_load.rename(columns={"Magnitude Type": "Magnitude_Type"})

    #preview df_load
    #print(df_load.head())

    #insert the records into the Quake model/table
    for index, row in df_load.iterrows():
        Date = row['Date']
        Latitude = row['Latitude']
        Longitude = row['Longitude']
        Type = row['Type']
        Depth = row['Depth']
        Magnitude = row['Magnitude']
        Magnitude_Type = row['Magnitude_Type']
        ID = row['ID']

        Quake(Date=Date, Latitude=Latitude, Longitude=Longitude, Type=Type, Depth=Depth, Magnitude=Magnitude, Magnitude_Type=Magnitude_Type, ID=ID).save()

if Quake_Predictions.objects.all().count() == 0:
    # Add the 2017 test data and the 1965 - 2016 training data
    df_test = pd.read_csv(r"C:\Users\bpraf_000\Downloads\earthquakeTest.csv")
    df_train = pd.read_csv(r"C:\Users\bpraf_000\Downloads\database.csv")

    df_train_load = df_train.drop(['Depth Error', 'Time', 'Depth Seismic Stations', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 
    'Root Mean Square', 'Source', 'Location Source', 'Magnitude Source', 'Status'], axis=1)

    df_test_load = df_test[['time', 'latitude', 'longitude', 'mag', 'depth']]

    df_train_load = df_train_load.rename(columns={'Magnitude Type': 'Magnitude_Type'})
    df_test_load = df_test_load.rename(columns={'time': 'Date', 'latitude': 'Latitude', 'longitude': 'Longitude', 'mag': 'Magnitude', 'depth': 'Depth'}) 

    # Create training and test dataframes
    df_test_data = df_test_load[['Latitude', 'Longitude', 'Magnitude', 'Depth']]
    df_train_data = df_train_load[['Latitude', 'Longitude', 'Magnitude', 'Depth']]

    # Remove all null values from both dataframes
    df_test_data.dropna()
    df_train_data.dropna()

    # Create features for the model out of the training data
    X = df_train_data[['Latitude', 'Longitude']]
    y = df_train_data[['Magnitude', 'Depth']]

    # Create test data features
    X_new = df_test_data[['Latitude', 'Longitude']]
    y_new = df_test_data[['Magnitude', 'Depth']]

    # Split our training features into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create random forest regressor model
    model_reg = RandomForestRegressor(random_state=50)
    # Train the model using the training data
    model_reg.fit(X_train, y_train)
    # Use the trained model to predict the training test data
    model_reg.predict(X_test)

    # Improve the model accuracy by automating hyperparameter tuning
    parameters = {'n_estimators': [10, 20, 50, 100, 200, 500]}
    # Create the gridsearchcv model
    grid_obj = GridSearchCV(model_reg, parameters)
    # Train the model using the training data
    grid_fit = grid_obj.fit(X_train, y_train)
    # Select the best fit model
    best_fit = grid_fit.best_estimator_
    # Use the best fit model to make the prediction on our training test data
    results = best_fit.predict(X_test)
    # Preview score
    score = best_fit.score(X_test, y_test) * 100
    #print(score)
    # Use the best fit model to make the predictions on our out of sample test data, earthquakes 2017
    final_results = best_fit.predict(X_new)
    # Evaluate the model accuracy
    final_score = best_fit.score(X_new, y_new) * 100
    # Store the prediction results into lists
    lst_Magnitude = []
    lst_Depth = []
    i = 0

    # Loop through our predicted magnitude and depth values to populate the lists
    for r in final_results.tolist():
        lst_Magnitude.append(final_results[i][0])
        lst_Depth.append(final_results[i][1])
        i += 1

    # Create our predicted earthquakes dataframe
    df_results = X_new[['Latitude', 'Longitude']]
    df_results['Magnitude'] = lst_Magnitude
    df_results['Depth'] = lst_Depth
    df_results['Score'] = final_score

    # Preview the prediction dataset
    #print(df_results.head())

    # Save model to postgreSQL
    for index, row in df_results.iterrows():
        Latitude = row['Latitude']
        Longitude = row['Longitude']
        Magnitude = row['Magnitude']
        Depth = row['Depth']
        Score = row['Score']

        Quake_Predictions(Latitude=Latitude, Longitude=Longitude, Magnitude=Magnitude, Depth=Depth, Score=Score).save()