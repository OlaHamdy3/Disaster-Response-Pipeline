# Disaster Response Pipeline Project

## Introduction

This repository houses an end-to-end solution for analyzing sentiments in messages sent during disaster events. The project includes text cleaning processes and a classification model designed to extract valuable insights.

The primary goal of this project is to provide a tool that aids in understanding the sentiments expressed in messages shared during disaster situations by leveraging natural language processing (NLP) techniques and machine learning models.

## File Structure
```
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app
|- transformer.py # imports transformer class

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # data cleaning code
|- DisasterResponse.db   # database to save clean data in

- models
|- train_classifier.py # model training code
|- transformer.py # transformer that adds text length to message features
|- classifier.pkl  # saved model 


- .gitignore # to ingore some files in git

- requirements.txt # required python packages

- README.md
```

## Instructions:
1. Install required packages
    - `pip install -r requirements.txt`

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Go to http://0.0.0.0:3000