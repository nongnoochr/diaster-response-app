import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Plot#1 - Show a number of messages of each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Plot#2 - Show a number of messages where the original language
    # is english (null value in the 'original' column) or 
    # non-english (the 'original' column is not null)
    num_non_en = df['original'].isnull().sum()
    num_en = df.shape[0] - num_non_en
    
    lang_names = ['English', 'Non-english']
    lang_counts = [num_en, num_non_en]
    
    # Plot#3 & 4 - Show a number of message in each category and 
    # use a color encoding for the genre
    
    # drop unused columns first
    df_cat_genre = df.drop(['id', 'message', 'original'], axis=1)
    
    # For the 'related' column, replace 2 with 0 since 2 means not related
    df_cat_genre['related'] = df_cat_genre['related'].replace(2, 0)
    
    # Get a list of category names sorted by a number of messages first!
    df_cat = df_cat_genre.drop(['genre'], axis=1) 
    cat_names = df_cat.sum().sort_values().index.tolist()
    
    # Then, group the dataframe by 'genre' and get a sum of each column
    # since 1 means related and 0 means not related
    df_cat_genre_count = df_cat_genre.groupby('genre').sum()
    
    # Ensure that the order of categories is sorted descendingly based on
    # a number of messages
    df_cat_genre_count = df_cat_genre_count[cat_names] 
    cat_genre_names = df_cat_genre_count.columns.tolist()
    
    # Array to a plot object for plots 3 and 4
    data_barplot = []
    data_dotplot = []
    
    # Use the same color coding for both plots 3 and 4
    genre_map = [
        {'name': 'social', 'color': 'rgba(0, 164, 0, 1.0)'},
        {'name': 'news', 'color': 'rgba(255, 118, 0, 1.0)'},
        {'name': 'direct', 'color': 'rgba(0, 120, 185, 1.0)'}
    ]
    for cur_genre_map in genre_map:
        # Get the current category name, color, and a number of messages
        cur_genre = cur_genre_map['name']
        cur_color = cur_genre_map['color']
        cur_val = df_cat_genre_count.loc[cur_genre,:].values.tolist()
        
        # Plot#3 - Horizontal stacked bar to show a total number of message 
        # in each category and use a color encoding for the genre
        data_barplot.append(
            Bar(
                y = cat_genre_names,
                x = cur_val,
                name= cur_genre,
                orientation = 'h',
            )
        )
        
        # Plot#4 - Dot plot to show a total number of message in each
        # category and use a color encoding for the genre
        data_dotplot.append(
            Scatter(
                y = cat_genre_names,
                x = cur_val,
                name= cur_genre,
                mode='markers',
                marker=dict(
                    color=cur_color,
                    line=dict(
                        color=cur_color,
                        width=1,
                    ),
                    symbol='circle',
                    size=12,
                )
            )
        )
    
    # create visuals
    graphs = [
        # Plot#1 - Show a number of messages of each genre
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
        
        # Plot#2 - Show a number of messages where the original language
        # is english or non-english
        {
            'data': [
                Bar(
                    x=lang_names,
                    y=lang_counts
                )
            ],

            'layout': {
                'title': 'The number of messages based on original',
                'yaxis': {
                    'title': "Count"
                }
            }
        },
        
        # Plot#3 - Horizontal stacked bar for a total number of message 
        # in each category
        {
            'data': data_barplot,

            'layout': {
                'barmode': 'stack',
                'title': 'Distribution of Message Categories by Genre',
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': "The number of messages"
                },
                'height': 800,
                'margin': dict(
                    l=250,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                ),
            }
        },
        
        # Plot#4 - Dot plot to show a total number of message in each
        # category 
        {
            'data': data_dotplot,

            'layout': {
                'title': 'The number of messages of each category and genre',
                'yaxis': {
                    'dtick': 1
                },
                'xaxis': {
                    'title': "The number of messages"
                },
                'height': 800,
                'hovermode': 'closest',
                'margin': dict(
                    l=250,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                ),
            }
        }
        
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()