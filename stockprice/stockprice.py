from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
if __name__ == "__main__":
    spark = SparkSession.builder\
        .master("local[*]")\
        .appName("stock_price").getOrCreate()

    data = spark.read.csv('data/stocks_price_final.csv', sep=',', header=True)
    # data.show()
    data.printSchema()

    # Select
    data.select('volume').show(5)
    data.select(['open', 'close', 'volume']).show(5)

    # Filter
    data.filter((col('high') >= lit('50')) & (col('low') >= lit('60'))).show(5)

    # When
    data.select('open', 'close',
                when(data.adjusted >= 59.6300, 1).otherwise(0).alias('is_adjusted')).show(5)

    # Group By
    data.select(['industry', 'open', 'close', 'adjusted'])\
        .groupBy('industry')\
        .mean()\
        .show(5)