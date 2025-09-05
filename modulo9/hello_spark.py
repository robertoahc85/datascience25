from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("HolaSpark")
    .master("local[*]")  # usa todos los cores locales
    .getOrCreate()
)

df = spark.createDataFrame([(1, "Hola"), (2, "Spark")], ["id", "msg"])
df.show()

spark.stop()
