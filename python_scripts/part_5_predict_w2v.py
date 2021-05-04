"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql.types import DateType

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Загрузка модели
loadedWord2Vec = Word2Vec.load(path='ml_models/word2vec_2021_05_04')
loadedModel = Word2VecModel.load(path='ml_models/word2vec-model_2021_05_04')
