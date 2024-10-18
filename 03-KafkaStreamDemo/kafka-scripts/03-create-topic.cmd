%KAFKA_HOME%\bin\windows\kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic invoices

docker exec kafka kafka-topics --create --topic invoices --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
