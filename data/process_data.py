"""
process_data.py
--------------------------------------------------
This script takes the raw data and clean it for later use in ML classifier training.
"""

# import libraries
import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Read the raw csv files and return merged data frame.

    Args:
        messages_filepath: messages input data file path
        categories_filepath: categories input data file path

    Returns:
        pd.DataFrame: merged data frame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    messages.head()

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()

    # merge datasets
    df = messages.merge(categories, on='id', how='inner')
    df.head()

    return df


def clean_data(df):
    """ Reads the data frame and clean the data for later use.

    Args:
        df: Dataframe containing the merged raw data.

    Returns:
        pd.DataFrame: Processed cleaned data frame.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.iloc[1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = list([x[:-2] for x in row])
    # print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames
    # categories.head()

    import copy
    c = copy.deepcopy(categories)
    for column in categories:
        # set each value to be the last character of the string
        #     print(categories[column])
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # categories.head()

    categories['related'] = categories['related'].apply(lambda x: 1 if x not in ['1', '0', 1, 0] else x)

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    # df.duplicated().sum()

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # check number of duplicates
    # df.duplicated().sum()

    return df


def save_data(df, database_filename):
    """ Save processed raw data into a SQlite database.

    Args:
        df: processed data
        database_filename: Database file name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('cleaned_data', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
