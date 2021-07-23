# Continuous Applications with Structured Streaming Python APIs in Apache Spark

This project shows how one can train a model using Apache Spark and MLlib then deploy that model using Spark's structured streaming for making predictions as a continunous application.

This application will use a credit card fraud use case to demonstrate how MLlib models and structured streaming can be combined, to constitutue a continunous application. In our hypothetical use case, we have some historical data of credit card transactions, some of which have been identified as fraud. We want to train a model using this historical data that can flag potentially fraudulent transactions coming in as a live stream. We then want to deploy that model as part of a data pipeline which will work with a stream of transaction data to identify potential fraud hotspots in a continunous manner.

