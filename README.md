# Disaster Response Pipeline Project

This project includes a web app that is using a Machine Learning model that will help an emergency worker to classify a message during diaster events in several categories and it also shows visualizations to show insights of the collecting data that is used to train the Machine Learning model.

This project also contains python scripts that is used in the ETL (Extract, Transform, and Load) and machine learning pipelines which generate a sqlite file that contains data after performing the ETL pipeline and a pickle file which stores a trained Machine Learning model that is used in the web app.

 

## Installation
Below are python libraries that are required to run this code using Python versions 3.*:

* numpy
* pandas
* sqlalchemy
* nltk
* sklearn

IMPORTANT: For nltk, ensure that 'punkt' and 'wordnet' are downloaded before running the script/app

```
import nltk

nltk.download('punkt')
nltk.download('wordnet')

```


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br />

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves () <br />

        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

        <p>
        Note that:

        1) the model is trained concurrently with the parameter n_jobs set to -1 (all CPUs are used)
        2) random_state=1 is used to train the model to ensure that the same generated model will be the same.
        3) This program can run quite long (hours) because it is also done GridSearch to find the best parameters in the provided list.
        </p>

2. Run the following command in the app's directory to run your web app. <br />
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements

Data files used in this project were from [Figure Eight](https://www.figure-eight.com/).
You can find the Licensing for the data and other descriptive information there.

This project is [MIT licensed](./LICENSE).