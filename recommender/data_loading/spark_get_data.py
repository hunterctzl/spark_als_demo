def load_csv_to_spark_df(path, spark):
    spark_df = spark.read.load(path, format='csv', header=True, inferSchema=True)
    return spark_df

def load_csv_to_spark_rdd(path, sc):
    spark_rdd = sc.textFile(path)
    return spark_rdd