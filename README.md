# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Motivation

For this project, I created a machine learning pipeline to categorize real messages that is sent during disaster events so that you can send the messages to an appropriate disaster relief agency.

The project include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Descriptions

There are three project components:

 1. ETL Pipeline: A Python script named `process_data.py`, which clean the data and stores it in a SQLite database (`DisasterResponse.db`).
 
 2. ML Pipeline: A Python script named `train_classifier.py`, which write a machine learning pipeline that:
 
       - Loads data from the SQLite database
       - Splits the dataset into training and test sets  
       - Builds a text processing and machine learning pipeline
       - Trains and tunes a model using GridSearchCV
       - Outputs results on the test set
       - Exports the final model as a pickle file (`classifier.pkl`)
       
 3. Flask Web App which adds data visualizations using Plotly in the web app. 
 
### Licensing, Authors, Acknowledgements

Must give credit to [Figure Eight](https://www.figure-eight.com/) for the data.

