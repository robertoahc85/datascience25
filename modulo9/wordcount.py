from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("WordCount").getOrCreate()

lines = spark.read.text("data/texto.txt")               # acción diferida
words = lines.select(F.explode(F.split(F.col("value"), r"\W+")).alias("w"))
clean = words.filter(F.col("w") != "")
counts = clean.groupBy("w").count().orderBy(F.desc("count"))

counts.show(20)  # ACCIÓN

spark.stop()
