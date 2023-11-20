import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT 
        messages_filepath - string with the path of disaster messages
        categories_filepath - string with the path of disaster categories

    OUTPUT
        Returns the messages and their categories
    '''
        
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')



def clean_data(df):
    '''
    INPUT 
        df - dataframe to be cleaned 

    OUTPUT
        Returns dataframe cleaned 
    '''
    categories = df['categories'].str.split(";", expand=True)
    category_colnames = list(categories.iloc[0].apply(lambda a: a.split('-')[0]))
    categories.columns = category_colnames

    for column in category_colnames:
        categories[column] = categories[column].str[-1] 
        categories[column] = pd.to_numeric(categories[column])
        
    df = df.drop(columns=['categories'])
    df = df.join(categories)
    
    df = df[df['related'].isin([1,2])]
    df = df.drop_duplicates()
    return df



def save_data(df, database_filename):
    '''
    INPUT 
        df - dataframe to be saved in database
        database_filename - string path of database

    OUTPUT
        Saves dataframe as a table in database
    '''
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')  


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