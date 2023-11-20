import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine
import joblib

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def count_categories_by_genre(df, genre_type=None):
    if genre_type:
        cat_count = df[df.genre == genre_type].drop(columns='genre').sum(axis=0).sort_values(ascending=False)
    else:
        cat_count = df.drop(columns='genre').sum(axis=0).sort_values(ascending=False)
    return cat_count.index, cat_count.values

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories = df.iloc[:, 3:]
    cat_names, cat_count = count_categories_by_genre(categories)
    direct_cat_names, direct_cat_count = count_categories_by_genre(categories, 'direct')
    news_cat_names, news_cat_count = count_categories_by_genre(categories, 'news')
    social_cat_names, social_cat_count = count_categories_by_genre(categories, 'social')

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=cat_names[:10],
                    y=cat_count[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_cat_names[:10],
                    y=direct_cat_count[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Categories of Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=social_cat_names[:10],
                    y=social_cat_count[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Categories of Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=news_cat_names[:10],
                    y=news_cat_count[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Categories of Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()