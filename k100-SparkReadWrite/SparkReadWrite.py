import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.streaming import StreamingQuery

if __name__ == "__main__":
    # Step 1: Create a Spark session
    spark = SparkSession.builder \
        .appName("Spark Read Write") \
        .master("local[*]").getOrCreate()
    # Create a directory for streaming input files
    input_dir = "D:\\tmp\\stream_input"
    os.makedirs(input_dir, exist_ok=True)


    # Step 2: Function to generate small CSV files (simulating data stream)
    def generate_files():
        for i in range(1, 100, 10):  # Create 10 records at a time (1 to 100)
            file_path = f"{input_dir}\\data_{i}.csv"
            with open(file_path, "w") as f:
                for j in range(i, i + 10):
                    f.write(f"{j}\n")
            time.sleep(1)  # Sleep to simulate new files arriving over time


    # Run the file generation in the background
    import threading

    file_thread = threading.Thread(target=generate_files)
    file_thread.start()

    # Step 3: Read the stream from the directory
    streaming_df = spark.readStream \
        .format("csv") \
        .option("header", "false") \
        .schema("number LONG") \
        .load(input_dir)

    # Step 4: Write the stream to the console for display
    write_query = streaming_df.writeStream \
        .format("console") \
        .outputMode("append") \
        .start()

    # Let the stream run for 5 seconds (simulated streaming)
    # write_query.awaitTermination(10) # Wait for 10 seconds
    spark.stop()
