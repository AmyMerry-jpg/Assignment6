
"""
loan_predict_pyspark.py

Objective:
Predict loan approval status using PySpark ML.
This script demonstrates a full end-to-end ML workflow including:
- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
"""

import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    Bucketizer
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Spark Session Initialization

# SparkSession is the entry point for PySpark functionality
spark = SparkSession.builder \
    .appName("LoanPredictionPySpark") \
    .getOrCreate()

# Load Data

base_dir = os.path.abspath(os.path.dirname(__file__))
input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"

df = spark.read.csv(input_path, header=True, inferSchema=True)

# Feature Selection

# Loan_ID is an identifier and has no predictive power → drop
df = df.drop("Loan_ID")

# Selected features rationale:
# - ApplicantIncome, CoapplicantIncome: financial strength
# - LoanAmount, Loan_Amount_Term: loan burden
# - Credit_History: strongest predictor of loan approval
# - Categorical demographics affect approval likelihood

# Handling Missing Values

# Categorical columns → replace missing values with "Unknown"
# Rationale: preserves row count and avoids bias from row deletion
categorical_cols = [
    "Gender", "Married", "Dependents",
    "Education", "Self_Employed", "Property_Area"
]

for c in categorical_cols:
    df = df.fillna({c: "Unknown"})

# Numerical columns → median imputation
# Rationale: median is robust to skewed income distributions
numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

for c in numerical_cols:
    median_value = df.approxQuantile(c, [0.5], 0.01)[0]
    df = df.fillna({c: median_value})

# Outliers

# Income variables are highly skewed → cap extreme values (Winsorization)
# Rationale: prevents models from being dominated by extreme salaries

income_cap = df.approxQuantile("ApplicantIncome", [0.99], 0.01)[0]
df = df.withColumn(
    "ApplicantIncome",
    when(col("ApplicantIncome") > income_cap, income_cap)
    .otherwise(col("ApplicantIncome"))
)

# Discretization

# Discretize LoanAmount into bins to capture non‑linear effects
# Rationale: Tree-based models benefit from bucketed numeric variables
splits = [-float("inf"), 100, 200, 300, float("inf")]

bucketizer = Bucketizer(
    splits=splits,
    inputCol="LoanAmount",
    outputCol="LoanAmountBucket"
)

df = bucketizer.transform(df)


# Label Encoding

# Convert Loan_Status to numerical label
df = df.withColumn(
    "label",
    when(col("Loan_Status") == "Y", 1).otherwise(0)
)


# Encode Categorical Variables

# StringIndexer converts categorical text values to numeric indices
# handleInvalid="keep" prevents crashes on unseen categories

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    ) for c in categorical_cols
]


# Feature Vector Assembly

# Combine all selected features into a single feature vector

feature_cols = (
    [f"{c}_idx" for c in categorical_cols] +
    numerical_cols +
    ["LoanAmountBucket"]
)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)


# Train-Test Split

# 80/20 split ensures unbiased evaluation
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)


# Model Definitions 

# Logistic Regression – strong baseline for binary classification
lr = LogisticRegression(
    featuresCol="features",
    labelCol="label"
)

# Random Forest – handles non-linearities and interactions well
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=8,
    seed=42
)

# Gradient-Boosted Trees – strong ensemble learner
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    seed=42
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Gradient Boosted Trees": gbt
}


# Training and Evaluation

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

for name, model in models.items():
    # Build pipeline: preprocessing + model
    pipeline = Pipeline(
        stages=indexers + [assembler, model]
    )

    # Train model
    fitted_model = pipeline.fit(train_df)

    # Generate predictions
    predictions = fitted_model.transform(test_df)

    # Evaluate accuracy
    accuracy = evaluator.evaluate(predictions)

    print(f"{name} Accuracy: {accuracy:.4f}")


# Stop Spark 

spark.stop()
