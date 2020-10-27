
def combine_data(data):
    data_recommend = data.drop(columns=['movie_id', 'original_title', 'plot'])
    data_recommend['combine'] = data_recommend[data_recommend.columns[0:2]].apply(
        lambda x: ','.join(x.dropna().astype(str)), axis=1)

    data_recommend = data_recommend.drop(columns=['cast', 'genres'])
    return data_recommend