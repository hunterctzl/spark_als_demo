def get_rating(movie_rating):
    header = movie_rating.take(1)[0]
    rating_data = movie_rating \
        .filter(lambda line: line!=header) \
        .map(lambda line: line.split(",")) \
        .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
        .cache()
    return rating_data

def data_split(data, ratio):
    train, validation, test = data.randomSplit(ratio, seed=99)
    return train, validation, test