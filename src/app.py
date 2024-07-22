# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pickle import dump
import os
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from dotenv import load_dotenv
import ast

# reading the datasets and saving it
movies = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv')
movies.to_csv('../data/raw/tmdb_5000_movies.csv', index=False)
movies.head()

credits = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv')
credits.to_csv('../data/raw/tmdb_5000_credits.csv', index=False)
credits.head()

# load the .env file variables
load_dotenv()

#Connect to the already created database here using the SQLAlchemy's create_engine function
def connect():
    global engine # Esto nos permite usar una variable global llamada "engine"
    # Un "connection string" es básicamente una cadena que contiene todas las credenciales de la base de datos juntas
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}?"
    print("Starting the connection...")
    engine = create_engine(connection_string)
    engine.connect()
    return engine

connect()

# load the tables into the database
movies.to_sql('movies', connect(), if_exists='replace', index=False)
credits.to_sql('credits', connect(), if_exists='replace', index=False)
print("Data has been successfully loaded into the database.")

# Create a new table for the merged data

# query = text("""
# CREATE TABLE merged_data AS
# (SELECT * FROM movies
# JOIN credits USING (title));
#              """)

# with engine.connect() as conn:
#     result = conn.execute()

# print("Table 'merged_data' has been created.")

# Load the merged data into a pandas DataFrame and saving it to a CSV file

merged_data = pd.read_sql_table('merged_data', engine.connect())
merged_data.head()
merged_data.to_csv('../data/raw/merged_data.csv', index=False)

# processing the relevant columns and saving the df we want to work with:
df = merged_data[['movie_id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']]
print(df.head())
df.to_csv('../data/processed/movies_mergedcleaned.csv', index=False)

# Convert the genre column from string to list of dictionaries
df['genres'] = df['genres'].apply(ast.literal_eval)
df['keywords'] = df['keywords'].apply(ast.literal_eval)
df['cast'] = df['cast'].apply(ast.literal_eval)

# defining function to extract names from json format
def extract_names(items):
    return ', '.join([item['name'] for item in items])

# applying name extraction funciton to the relevant columns
df['genre_names'] = df['genres'].apply(extract_names)
df.drop(columns=['genres'], inplace=True)
df['keywords_names'] = df['keywords'].apply(extract_names)
df.drop(columns=['keywords'], inplace=True)
df['cast_names'] = df['cast'].apply(extract_names)
df.drop(columns=['cast'], inplace=True)

# renaming columns for better readability
df.rename(columns={'genre_names': 'genres', 'keywords_names': 'keywords', 'cast_names': 'cast'}, inplace=True)

# splitting the text in the columns:
df['genres'] = df['genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
df['keywords'] = df['keywords'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
df['cast'] = df['cast'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

# converting the columns crew (json) to list and overwiew (plain text) to list:
def convert_to_list(item):
    if isinstance(item, str):
        try:
            return ast.literal_eval(item)
        except:
            return []
    elif item is None:
        return []
    else:
        return item
    
df['crew'] = df['crew'].apply(convert_to_list)


def text_to_list(text):
    if isinstance(text, str):
        return [text]
    else:
        return []

df['overview'] = df['overview'].apply(text_to_list)

# Extracting first three cast names:
def extract_first_three(cast_list):
    if len(cast_list) > 3:
        return cast_list[:3]
    else:
        return cast_list
    
df['cast'] = df['cast'].apply(extract_first_three)

# Extracting director's name:
def extract_director(crew_list):
    return [member['name'] for member in crew_list if member['job'] == 'Director']

# storing it in a new column and deleting the original crew column:
df['director'] = df['crew'].apply(extract_director)
df = df.drop('crew', axis=1) 

# Combine all columns into 'tags' column
def combine_to_tags(row):
    parts = [] # creating a list to store all pieces of text to be combined
     # checking if the column is a list, and if so add it to 'parts'
    if isinstance(row['overview'], list):
        parts.extend(row['overview'])
    if isinstance(row['genres'], list):
        # for all other columns, remove spaces between words before adding them to 'parts'
        row['genres'] = [i.replace(" ", "") for i in row['genres']]
        parts.extend(row['genres'])
    if isinstance(row['cast'], list):
        row['cast'] = [i.replace(" ", "") for i in row['cast']]
        parts.extend(row['cast'])
    if isinstance(row['director'], list):
        row['director'] = [i.replace(" ", "") for i in row['director']]
        parts.extend(row['director'])
    
    return parts
# creating the column tags and applying the function
df['tags'] = df.apply(combine_to_tags, axis=1)

# now joining all the elements of parts into a single string
df["tags"] = df["tags"].apply(lambda x: ",".join(x).replace(",", " "))
# removing the original columns:
df.drop(columns = ["genres", "keywords", "cast", "director", "overview"], inplace = True)

print(df['tags'][0]) # checking if it worked
print(df.head())

# saving the data
df.to_csv('../data/processed/final_data.csv', index=False)

# VECTORIZING THE TAGS
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['tags'])

# APPLYING COSINE SIMILARITY (KNN) TO THE TAGS
similarity = cosine_similarity(X)

# DEFINING THE RECOMMENDER FUNCTION:
new_df = df.copy()

def recommend(movie):
    # get the index of the movie we want to find similar movies to
    movie_index = new_df[new_df["title"] == movie].index[0]
    # finding the nearest cosine similar movies
    distances = similarity[movie_index]
    # create the recommendation list of movies
    movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# asking the user for a movie to find recommendations
film = str(input("Enter a movie to find recommendations:  "))
film = film.lower()
recommend(film)
