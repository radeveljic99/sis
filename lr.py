from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_extract
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Titanic").getOrCreate()

    # Ucitavanje test i train cvs fajlova
    train_df = spark.read.csv("train.csv", header=True, inferSchema=True)
    test_df = spark.read.csv("test.csv", header=True, inferSchema=True)

    # Racunamo srednju vrijednost za kolonu godine i dje je null popunimo sa srednjom vrijednoscu
    mean_age = train_df.agg({"Age": "mean"}).collect()[0][0]
    train_df = train_df.fillna({'Age': mean_age, 'Embarked': 'S'}) # Embarked zamjenujemo sa S dje ne postoji jer je vecina u Southampton
    test_df = test_df.fillna({'Age': mean_age, 'Embarked': 'S'})

    # Radimo castovanje kolona sa pravim vrijednostima
    train_df = train_df.withColumn("Pclass", col("Pclass").cast("double")) \
                       .withColumn("SibSp", col("SibSp").cast("double")) \
                       .withColumn("Parch", col("Parch").cast("double")) \
                       .withColumn("Fare", col("Fare").cast("double"))

    test_df = test_df.withColumn("Pclass", col("Pclass").cast("double")) \
                     .withColumn("SibSp", col("SibSp").cast("double")) \
                     .withColumn("Parch", col("Parch").cast("double")) \
                     .withColumn("Fare", col("Fare").cast("double"))


    # Nakon castovanja i za ostale vrijednosti popunjavamo dje je null ili None
    train_df = train_df.fillna({'Pclass': 3, 'Fare': train_df.agg({"Fare": "mean"}).collect()[0][0], 'SibSp': 0, 'Parch': 0})
    test_df = test_df.fillna({'Pclass': 3, 'Fare': test_df.agg({"Fare": "mean"}).collect()[0][0], 'SibSp': 0, 'Parch': 0})

    # pol pretvaramo u 0 i 1
    train_df = train_df.withColumn("Sex", when(col("Sex") == "male", 1).otherwise(0))
    test_df = test_df.withColumn("Sex", when(col("Sex") == "male", 1).otherwise(0))

    # Preprocesiranje kolone Cabin
    # Ekstrahovanje prvog karaktera kabine kao tip kabine (npr. 'C', 'B', 'D')
    train_df = train_df.withColumn("Cabin", regexp_extract(col("Cabin"), '([A-Za-z])', 1))
    test_df = test_df.withColumn("Cabin", regexp_extract(col("Cabin"), '([A-Za-z])', 1))


    # Zamena null kabina sa najčešćim tipom kabine (npr. 'C')
    filtered_train_df = train_df.filter(col("Cabin").isNotNull())

    # Find the most common cabin from the filtered dataset
    most_common_cabin_row = filtered_train_df.groupBy("Cabin").count().orderBy(col("count").desc()).first()
    most_common_cabin = most_common_cabin_row[0] if most_common_cabin_row else "C"  # Default to 'C' if no common cabin found

    train_df = train_df.fillna({'Cabin': most_common_cabin})
    test_df = test_df.fillna({'Cabin': most_common_cabin})

    # Pretvaranje tipova kabine u numerički format koristeći StringIndexer
    indexer = StringIndexer(inputCol="Cabin", outputCol="CabinIndex")

    train_df = indexer.fit(train_df).transform(train_df)
    test_df = indexer.fit(test_df).transform(test_df)

    # Prebacanje datih kolona u features vektor
    assembler = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "CabinIndex"],
        outputCol="features"
    )

    # Transformisanje
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    # Treniranje LR modela, oznacavamo kolonu za koju radimo prediction i featuresNiz
    lr = LogisticRegression(labelCol="Survived", featuresCol="features")
    model = lr.fit(train_df)

    # model.transform vraca novi dataframe koji cuvamo u predictions df
    train_predictions = model.transform(train_df)

    # Testiranje na training_set
    evaluator = BinaryClassificationEvaluator(labelCol="Survived")
    train_accuracy = evaluator.evaluate(train_predictions)
    print(f"Training Accuracy: {train_accuracy}")

    # Predvidjanje nad testnim df
    test_predictions = model.transform(test_df)

    # Izdvajamo kolone potrebne za submission
    submission_df = test_predictions.select("PassengerId", "prediction")

    #  Preimenovanje prediction u survived
    submission_df = submission_df.withColumnRenamed("prediction", "Survived")

    # Radimo castovanje u int i micemo sve kolone koje eventualno imamo da su null ili None
    submission_df = submission_df.withColumn("Survived", col("Survived").cast("integer"))
    submission_df = submission_df.na.drop()

    # cuvanje u csv ( ovdje se napravi vise fajlova, jer moze da ima vise particija)
    submission_df.write.csv('submission-lr', header=True, mode="overwrite")
    spark.stop()
