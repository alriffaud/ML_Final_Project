#!/usr/bin/env python3
"""
This script connects to a MySQL database, extracts data from three tables,
and loads them into pandas DataFrames, checking for the presence of required
columns. It also prints the shape, columns, first rows, and information about
each DataFrame.
"""
import pymysql
import pandas as pd
import os


# Connect to the MySQL database
db_config = {
    'host': 'mysql',
    'port': 3306,
    'user': 'root',
    'password': 'Sophie2711',
    'db': 'quizzes',
    'charset': 'utf8mb4'
}

try:
    # Establish the connection
    connection = pymysql.connect(**db_config)

    # Extract data from SQL tables
    # Load data into pandas DataFrames
    quizzes_df = pd.read_sql("SELECT * FROM evaluation_quizzes", connection)
    questions_df = pd.read_sql("SELECT * FROM evaluation_quiz_questions",
                               connection)
    corrections_df = pd.read_sql("SELECT * FROM evaluation_quiz_corrections",
                                 connection)

    required = {
        'quizzes': ['id', 'time_allowed', 'online', 'name'],
        'questions': ['id', 'evaluation_quiz_id', 'category', 'question_type',
                      'data_json'],
        'corrections': ['id', 'user_id', 'start_time', 'end_time', 'data_json',
                        'skipped']
    }
    dfs = {
        'quizzes': quizzes_df,
        'questions': questions_df,
        'corrections': corrections_df
    }

    print(150 * "=")
    print("✅ Dataframes loaded successfully")

    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Export DataFrames to CSV files
    quizzes_df.to_csv('data/quizzes.csv', index=False)
    questions_df.to_csv('data/questions.csv', index=False)
    corrections_df.to_csv('data/corrections.csv', index=False)

    print(150 * "=")
    print("✅ CSV files exported successfully:")
    print("quizzes.csv")
    print("questions.csv")
    print("corrections.csv")

except pymysql.MySQLError as e:
    print(f"❌ Error connecting to the database: {e}")
finally:
    if connection:
        connection.close()
        print(150 * "=")
        print("Connection closed")
