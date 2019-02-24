import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load the specified message and categories data and merge them using the 'id'
    column. The output is a merged data frame.

    Args:
    messages_filepath: string. A relative path to the message data file.
    categories_filepath: string. A relative path to the categories data file.

    Returns:
    df: dataframe. A merged data frame.

    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id', how='outer')
    return df
    
    


def clean_data(df):
    '''
    Clean the categories data of the specified dataframe

    Args:
    df: dataframe. Merged dataframe before cleaning.

    Returns:
    df: dataframe. The cleaned dataframe.

    '''

    # === Split categories into separate category columns

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-', expand=True)[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # === Convert category values to just numbers 0 or 1
    for column in category_colnames:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # === Replace categories column in df with new category columns

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # === Remove duplicates

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Save the dataframe to a sqlite file.

    Args:
    df: dataframe. Cleaned dataframe with data to be saved to a sqlite file
    database_filename: string. Sqlite file name

    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    engine.execute('DROP TABLE IF EXISTS InsertTableName')
    df.to_sql('InsertTableName', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()