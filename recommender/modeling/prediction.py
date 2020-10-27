from recommender.data_loading.get_data import get_data
from recommender.data_cleaning.combine import combine_data
from recommender.feature_engineering.transform import transform_data
from recommender.modeling.recommend import recommend_movies

def results(movie_name):
    movie_name = movie_name.lower()

    find_movie = get_data()
    combine_result = combine_data(find_movie)
    transform_result = transform_data(combine_result, find_movie)

    if movie_name not in find_movie['original_title'].unique():
        return 'Movie not in Database'

    else:
        recommendations = recommend_movies(movie_name, find_movie, combine_result, transform_result)
        return recommendations.to_dict('records')