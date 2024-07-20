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
from pickle import dump
import os
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from dotenv import load_dotenv

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

