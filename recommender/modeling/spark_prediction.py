from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS
from recommender.data_loading.spark_get_data import load_csv_to_spark_df, load_csv_to_spark_rdd
from recommender.data_cleaning.spark_preprocessing import get_rating, data_split
from recommender.modeling.spark_collaborative_filtering import als_gridsearch
from recommender.modeling.spark_recommend import make_recommendation, make_recommendation_new

# spark config
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "96g") \
    .config("spark.driver.memory", "96g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()
# get spark context
sc = spark.sparkContext

def results(user_id):
    
    # load data
    movies = load_csv_to_spark_df('dataset/movies.csv', spark)
#     movie_rating = load_csv_to_spark_rdd('dataset/ratings.csv', sc)
    movie_rating = load_csv_to_spark_rdd('dataset/ratings_greater_500.csv', sc)
    
    # preprocess data -- only need ["userId", "movieId", "rating"]
    rating_data = get_rating(movie_rating)
    
    if user_id not in rating_data.map(lambda r: r[0]) \
    .distinct() \
    .collect():
        return 'User not found'
    
    else:      
        # train model
        model = ALS.train(ratings=rating_data, iterations=10, rank=14, lambda_=0.05, seed=99)
        
        # get recommends
        recommends = make_recommendation(
            model = model, 
            ratings_data = rating_data, 
            df_movies = movies, 
            user_id = user_id, 
            n_recommendations = 10, 
            spark_context = sc)

        recommendations = {}
        for i, title in enumerate(recommends):
            recommendations[i+1] = title
        return recommendations