# Disaster Response Pipeline Project

### Introduction
This project is done under the Udacity Data Science Nano Degree.  
The goal of this project is to classify different messages, that people
in a distress during any disaster might send, into proper classes for
easy support and help can planned and assisted.

### Installation <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing and Acknowledgements <a name="licensing"></a>
This data set used for training is provided by [Figure Eight](https://www.figure-eight.com/)  
and Udacity. This repository is given under the MIT license.  
A copy of the license can be found at [here.](https://github.com/vasthav/DSNN-disaster-response-pipelie-project/blob/master/LICENSE)