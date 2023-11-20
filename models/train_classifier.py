import sys
import pickle
import re
import pandas as pd

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - string path of databse 

    OUTPUT
        X - Series features of traing data, messages
        y - Series label of trainging data
        y_cols - list of label names
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    df = pd.read_sql_table('disaster_messages', conn) 
    X = df['message']
    y_cols = list(set(df.columns.values) - set( ['id','message', 'original', 'genre']))
    y = df[y_cols]
    return X, y, y_cols


def tokenize(text):
    '''
    INPUT 
        text - string 

    OUTPUT
        tokens - list of lemmatized tokens
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text.lower())
    
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens if token not in stop_words]
    return tokens


def build_model():
    '''
    INPUT 
        None 

    OUTPUT
        model - classification model
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
    (('tfidf', TfidfTransformer())),
    ('model', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'tfidf__smooth_idf': [True, False],
    'model__estimator__max_depth': [3, 5, 7],
    'model__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT 
        model - classification model 
        X_test - features of testing data 
        Y_test - labels of testing data 
        category_names - labels names 

    OUTPUT
        prints precision, recall and accuracy - evaluates the model with testing data
    '''
    tot_acc = 0
    Y_pred = model.predict(X_test)

    for i, cat in enumerate(category_names):    
        metrics =  classification_report(Y_test[Y_test.columns[i]], Y_pred[:,i])
        tot_acc += accuracy_score(Y_test[Y_test.columns[i]], Y_pred[:,i])
        print(cat, 'accuracy: {:.3f}'.format(accuracy_score(Y_test[Y_test.columns[i]], Y_pred[:,i])))
        print(metrics)
    print('total accuracy {:.3f}'.format(tot_acc/len(category_names)))


def save_model(model, model_filepath):
    '''
    INPUT 
        model - classification model 
        model_filepath - string path to save model 
    OUTPUT
        saves model in file
    '''
    pickle.dump(model, open(model_filepath, "wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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