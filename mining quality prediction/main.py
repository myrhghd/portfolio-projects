from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow
import mlflow.spark

# Initialize Spark session
spark = SparkSession.builder.appName("######").getOrCreate()

# Load the best model from the local path
model_uri = "best_model"  # Local path to the best model directory
best_model = mlflow.spark.load_model(model_uri)

# Load the validation data
validation_data = spark.read.csv('Data/mining_validation_data.csv', header=True, inferSchema=True)

# Assemble features
feature_columns = [col for col in validation_data.columns if col != '% Iron Concentrate']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
validation_data = assembler.transform(validation_data).select('features', '% Iron Concentrate')

# Make predictions
predictions = best_model.transform(validation_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol='% Iron Concentrate', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f'Validation RMSE: {rmse}')

# Show a few predictions
predictions.select('features', 'prediction', '% Iron Concentrate').show()
