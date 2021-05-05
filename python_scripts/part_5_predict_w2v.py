"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec, Word2VecModel

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

# Загрузка модели
loadedWord2Vec = Word2Vec.load(path='ml_models/word2vec_2021_05_04')
loadedModel = Word2VecModel.load(path='ml_models/word2vec-model_2021_05_04')


def my_synonyms(model, word, num_synonyms=5):
    return model \
        .findSynonyms(word=word, num=num_synonyms) \
        .select('word').withColumnRenamed(existing='word', new='similar for {}'.format(word))


loadedModel.getVectors().orderBy('word').show(n=100, truncate=False)


required_products = ['(15835) Бинт марл мед стер 5м х 10см уп N1 671',
					 '(179313) Жасмин масло эфирное 10мл 410',
					 '(100178) Компливит Хондро тб №30 644']

for p in required_products:
	my_synonyms(model=loadedModel, word=p, num_synonyms=3).show(truncate=False)
