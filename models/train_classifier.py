import sys
import re
import pickle
import datetime
import multiprocessing


import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

def load_data(database_filepath):
    '''
    Load data from the specified database

    Args:
    database_filepath: string. A relative path to the database file

    Returns:
    X: Array of features data which is data in the 'message' column
    y: Array of labels data which is the 36 categories in the dataset
    category_names: List of category names corresponding to columns of y
    '''
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('InsertTableName', engine)
    # Feature variable X is data in the 'message' column
    X = np.array(df['message'])

    cat_values = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # Target variable Y is the 36 categories in the dataset
    if pd.__version__.startswith('0.24'):
        Y = cat_values.to_numpy()
    else:
        Y = cat_values.values

    category_names = cat_values.columns.tolist()

    return X, Y, category_names



def get_wordnet_pos(tag):
    ''' 
    Get a TreeBank tag from the specified WordNet part of speech name

    Args:
    tag: string. WordNet part of speech name.

    Returns:
    A corresponding TreeBank tag
    '''

    treebank_tag = ''
    # Refer to 
    # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

    if tag.startswith('J'):
        # Adjective
        treebank_tag = wordnet.ADJ

    elif tag.startswith('V'):
        # Verb
        treebank_tag = wordnet.VERB

    elif tag.startswith('N'):
        # Noun
        treebank_tag = wordnet.NOUN

    elif tag.startswith('R'):
        # Adverb
        treebank_tag = wordnet.ADV

    else:
        # Use Noun as a default output if none of above matches
        treebank_tag = wordnet.NOUN

    return treebank_tag

def tokenize(text):
    '''
    Perform a tokenization process on the input text

    Args:
    text: string. A message to be tokenized

    Returns:
    clean_tokens
    '''

    # Case normalization
    text = text.lower()
    
    # Punctuation Removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # --- Tokenization
    words = word_tokenize(text)
    
    # --- Remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
    
    # --- Part of speech tagging
    pv_tags = pos_tag(words)
    
    # --- Lemmatization
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for cur_tag in pv_tags:
        
        # Get current text
        cur_text = cur_tag[0]

        # Get a corresponding part of speech that will be used with the lemmatizer
        w_tag = get_wordnet_pos(cur_tag[1])
        
        # lemmatize the text with pos and append it to clean_tokens
        clean_tok = lemmatizer.lemmatize(cur_text, w_tag)
        clean_tokens.append(clean_tok)
    
    return clean_tokens
    

def build_model():
    '''
    Create a GridsearchCV object of a pipeline where the MultiOutputClassifier 
    is used along with the RandomForestClassifier as an estimator

    Returns:
    model: GridsearchCV object.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': (20, 50),
    }

    model = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Run the predict method of the specified model with a given input data and
    print out the best parameter found by the model (GridSearchCV object) and also
    report the f1 score, precision and recall for each output category of the dataset

    Args:
    model: A trained GridSearchCV object
    X_Test: Array of feature's test data
    Y_Test: Array of label's test data
    category_names: List of category names corresponding to each column in Y_Test
    '''

    # Predict labels of the feature's test data
    print('{} : Start model.predict'.format(datetime.datetime.now()))
    Y_pred = model.predict(X_test)
    print('{} : Finish model.predict'.format(datetime.datetime.now()))

    # Print out the best parameters found by GridSearch
    print('Best parameters:\n{}'.format(model.best_params_))  

    # Print out the f1 score, precision and recall for each output category of the dataset
    for index, col_name in enumerate(category_names):
        print('Column#{} - {}'.format(index, col_name))
        print(classification_report(Y_test[:, index], Y_pred[:, index]))


def save_model(model, model_filepath):
    '''
    Save a model to a pickle file at the speicified file path

    Args:
    model: A model object
    model_filepath: A relative path of the output file path
    '''

    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print('{} : Start model.fit'.format(datetime.datetime.now()))
        model.fit(X_train, Y_train)
        print('{} : Finish model.fit'.format(datetime.datetime.now()))
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()