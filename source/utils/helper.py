from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils


def config_spark_dbutils() -> (SparkSession, DBUtils):
    spark = SparkSession.builder.getOrCreate()
    dbutils = DBUtils(spark)
    return spark, dbutils
