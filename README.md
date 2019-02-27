# Disaster Response Pipeline Project

This project includes a web app that is using a Machine Learning model that will help an emergency worker to classify a message during diaster events in several categories and it also shows visualizations to show insights of the collecting data that is used to train the Machine Learning model.

This project also contains python scripts that is used in the ETL (Extract, Transform, and Load) and machine learning pipelines which generate a sqlite file that contains data after performing the ETL pipeline and a pickle file which stores a trained Machine Learning model that is used in the web app.

 

### Installation
Below are python libraries that are required to run this code using Python versions 3.*:

* numpy
* pandas
* sqlalchemy
* nltk
* sklearn

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements

Data files used in this project were from [Figure Eight](https://www.figure-eight.com/).
You can find the Licensing for the data and other descriptive information there.

This project is [MIT licensed](./LICENSE).