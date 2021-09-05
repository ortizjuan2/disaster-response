import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ load csv files with messages and categories,
        create a dataset merging both files with id as key

    input:
        messages_filepath: str, path to messages csv file
        categories_filepath: str, path to categories csv file

    return:
        df: dataframe with data loaded
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # concat both files
    df = messages.merge(categories, on='id', how='left')
    df = pd.concat([df,
        df['categories'].str.split(';', expand=True)], axis='columns')
    return df


def clean_data(df):
    """ clean dataset, convert categories to numeric values
        and ensure categories only have two values (0, 1)

    input:
        df: dataframe with data

    return:
        df: cleaned dataframe
    """
    # first row as column's name for categories
    df.columns = list(df.columns[:5]) + [c.split('-')[0] 
        for c in df.iloc[0, 5:]]

    categories = df.iloc[:, 5:]

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda s: int(s[-1]))
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # ensure only 0 or 1 is present in data
        categories.at[categories[column] > 0, column] = 1
    # drop original categories
    df = df.iloc[:, :4]

    df = pd.concat([df, categories], axis='columns')

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """ save the clean dataset to a sqlite database

    input:
        df: dataframe with the data
        database_filename: str, database file path
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('categories', engine, index=False)


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
