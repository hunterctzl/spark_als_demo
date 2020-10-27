import pandas as pd

def get_data():
        movie_data = pd.read_csv('dataset/movie_data.csv.zip')
        movie_data['original_title'] = movie_data['original_title'].str.lower()
        return movie_data