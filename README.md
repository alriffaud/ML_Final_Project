<h1 align="center">Machine Learning - Final Project</h1>

<p align="center">
  <img src="https://i.imgur.com/9pbFvRz.png">
</p>

## Table of Contents
- [Project overview:](#Project-overview)
- [System Architecture](#System-Architecture)
- [Technologies Used](#Technologies-Used)
- [Data Preprocessing](#Data-Preprocessing)
  - [Obtaining data files](#Obtaining-data-files)
  - [Data Extraction & Schema Validation](#Data-Extraction-&-Schema-Validation)
  - [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis-(EDA))
  - [Extract-Transform-Load (ETL) Pipeline](#Extract-Transform-Load-(ETL)-Pipeline)
  - [Data Splitting: Train, Validation, Test](#Data-Splitting:-Train,-Validation,-Test)
- [Knowledge Tracing Models](#Knowledge-Tracing-Models)
  - [Bayesian Knowledge Tracing (BKT)](#Bayesian-Knowledge-Tracing-(BKT))
  - [Performance Factors Analysis (PFA)](#Performance-Factors-Analysis-(PFA))
  - [Deep Knowledge Tracing (DKT)](#Deep-Knowledge-Tracing-(DKT))
  - [Self-Attentive Knowledge Tracing (SAKT)](#Self-Attentive-Knowledge-Tracing-(SAKT))
  - [Attentive Knowledge Tracing (AKT)](#Attentive-Knowledge-Tracing-(AKT))
- [Comparative Analysis Models](#Comparative-Analysis-Models)
- [SAKT Inference Application](#SAKT-Inference-Application)
- [Challenges, Lessons Learned & Future Directions](#Challenges,-Lessons-Learned-&-Future-Directions)
- [Conclusion](#Conclusion)


## Project overview:

The primary goal of this capstone project was to design, implement, and rigorously compare multiple Knowledge Tracing (KT) models — ranging from classical to state‑of‑the‑art transformer‑based architectures — and then finalize a simple implementation based on the best model obtained. This was achieved using real educational data from the Holberton School platform.

By building a full pipeline that spans from raw MySQL data extraction to model training and evaluation, this project sought to:
1. **Ingest and preprocess** large-scale quiz interaction data with robust cleaning, feature engineering, and balanced train/validation/test splits.
2. **Implement five KT models** — Bayesian Knowledge Tracing (BKT), Performance Factor Analysis (PFA), Deep Knowledge Tracing (DKT), Self‑Attentive Knowledge Tracing (SAKT), and Attentive Knowledge Tracing (AKT).
3. **Evaluates** each model on a held‑out test set, analyzing aggregate metrics like AUC and accuracy, calibration, attention interpretability, and per‑student learning trajectories.
4. **Compare** models to determine which architecture offers the best balance of predictive performance, interpretability, computational efficiency, and ease of deployment in a real educational platform.
5. **Apply** the model that shows the best results by performing a small implementation.
6. **Identify limitations** and chart next steps for advancing personalized learning recommendations in an educational platform.

Educational platforms strive to personalize learning and help students focus on the topics where they struggle most. By accurately modeling how and when a student masters a concept, instructors and adaptive systems can:

- Target interventions: Recommend focused reviews on weak topics immediately after a quiz.
- Optimize pacing: Decide whether to advance or remediate based on predicted readiness for the next activity.
- Measure learning gains: Go beyond raw scores and see how a student’s knowledge trace evolves over time.

This project demonstrates end‑to‑end how to build, compare, and deploy state‑of‑the‑art Knowledge Tracing models, culminating in a SAKT‑based inference application, which together form a robust framework for data‑driven, personalized education.

## System Architecture

Below is a high‑level diagram of the complete solution, from data ingestion to real‑time inference:
<p align="center">
  <img src="https://i.imgur.com/BIZS8Iq.png">
</p>
<p align="center">
  <img src="https://i.imgur.com/tBBrqXC.png">
</p>
<p align="center">
  <img src="https://i.imgur.com/nS90o7y.png">
</p>

## Technologies Used

The following is a table showing the main technologies used during the project, indicating the use given to each:
<p align="center">
  <img src="https://i.imgur.com/VBPbRt9.png">
</p>

## Data Preprocessing

### [Obtaining data files](#Data_Preprocessing/export_to_CSV.py)

Note: This part is optional, as the output files are already available in the Data_Preprocessing/data folder.

First, start by extracting the **evaluation_quizzes**, **evaluation_quiz_questions** and **evaluation_quiz_corrections** tables from the *quizzes.sql* file. To do this, the following command is executed in the terminal, after having installed and configured MySQL, and being located in the main folder of the project:

```bash
$ echo "CREATE DATABASE quizzes;" | mysql -uroot -p
Enter password:
$ cat Data_Preprocessing/data/quizzes.sql | mysql -uroot -p
Enter password:
```	
This will create the quizzes database and load the quizzes.sql file into it. Next, to run the Python script, you must have Python 3.8 or higher and the required libraries installed. To install the required libraries, run the following command in the terminal:

```bash
$ pip install -r requirements.txt
```
Once the libraries are installed, the Python script (file [export_to_CSV.py](#Data_Preprocessing/export_to_CSV.py)) can be executed with the following command from the Data_Preprocessing folder:

```bash
$ python3 export_to_CSV.py
```

This will generate the output files in the Data_Preprocessing/data folder.

### [Data Extraction & Schema Validation](#Data_Preprocessing/1_data_extraction.ipynb)

If you are using Colab, you should copy the *quizzes.csv*, *questions.csv* and *corrections.csv* files into a folder called *data* in the root directory of the project. If you cloned the repository and are working locally, you will already have the folder with the necessary files. In addition (in case you are using Colab) you should copy the file requirements.colab.txt into the root directory of the project. This file contains the libraries needed to run the *1_data_extraction.ipynb* script (you must uncomment the first cell in order to install the required libraries.). In case you are working locally, you can skip this step, as the requirements.txt file is already included in the repository. To install the required libraries, run the following command in the terminal:

```bash
$ pip install -r requirements.colab.txt
```
Once you have installed the necessary libraries, you can run the *1_data_extraction.ipynb* script in Jupyter Notebook or Google Colab. This script will extract the data from the CSV files and perform schema validation. Make sure the CSV files are in the *data* folder before running the script.

You can open the *1_data_extraction.ipynb* file to see the contents in detail and the steps performed in the script. The following is a brief description of the steps performed in it:
1. **Load DataFrames**: The necessary libraries are imported and the CSV files are loaded into pandas dataframes.
2. **Schema Validation (Key Columns)**: It is verified that the key columns are present in the dataframes.
3. **Quick Data Inspection**: In this section we perform the following steps:
    - print the shape of each DataFrame to know how many rows and columns we have.
    - check the columns of the dataframes to ensure they are as expected.
    - check the first rows of the dataframes to know the data better.
    - obtain the information about the dataframes to check the types of the columns and their null values.

### [Exploratory Data Analysis (EDA)](#Data_Preprocessing/2_eda.ipynb)

Once the data has been extracted and validated, we can proceed to the exploratory data analysis (EDA) phase. This phase is performed in the *2_EDA.ipynb* script. In this script, we will analyze the data to gain insights and understand its characteristics.

Again we need, in case we are in Colab, to copy the *requirements.colab.txt* file to the root of the project, together with the CSV files. 

The following steps are performed in this script:
1. **Load DataFrames**: the necessary libraries are imported and the CSV files are loaded into pandas dataframes.
2. **Basic Counts**: count the number of quizzes, questions, corrections, and unique students in the corrections DataFrame.
3. **Missing & Duplicates**: check for missing values and duplicates in the DataFrames.
4. **Category Distribution and Questions with Missing Category**: plot the distribution of categories in the quizzes DataFrame and analyze the uncategorized questions to better understand which questions to consider and which not.
5. **Questions whose *question_type* field is *Input***: analyze all those questions whose category is not *unknown*, and whose value of the *question_type* field is *input*.
6. **Real interaction counts per skill**: analyze the total number of interactions and unique students for each category.
7. **Answers to questions not found in the **questions_df** dataframe**: analyze whether there are answers to questions, where the *id* of the same does not correspond to any question in the dataframe **questions_df**.
8. ***star_time* and *end_time* fields NULL**: analyze the *star_time* and *end_time* fields in the **corrections_df** dataframe where the values ​​are NULL.
9. **Cases without the *question_answers* key in the *data_json* field**: analyze the corrections that do not have the *question_answers* key in the *data_json* field of the corrections_df dataframe.
10. **Response Durations**: in this section we perform the following steps:
    - analyze the null values ​​in the *start_time* and *end_time* fields of the **corrections_df** dataframe and determine the minimum and maximum duration of the quizzes in seconds.
    - explore the distribution of the response times to see if there are any outliers. We will use an histogram to visualize the distribution and identify any potential outliers.
    - **Winsorize**: cap the values ​​to the 99th percentile.
    - identify the values that were capped and display them.
    - analyze if there is a direct correlation between the difference in the values ​​of the columns duration_min and time_allowed_min and between the values ​​of the column days_offset.
    - create a new column called estimated_duration in the **corrections_df** dataframe
11. **Hit rates by correction, question and category**: in this section we perform the following steps:
    - calculate the hit rates for each correction.
    - calculate the hit rates for each question.
    - verify the hypothesis that if the *hit_rate* is equal to zero, then the *value* field is not observed in the *items* list of the *data_json* dictionary.

<p align="center">
  <img src="https://i.imgur.com/9Gskyly.png">
</p>

### [Extract-Transform-Load (ETL) Pipeline](#Data_Preprocessing/3_etl.ipynb)
The ETL pipeline is performed in the *3_etl.ipynb* script. In this script, we will extract the data from the CSV files, transform it to fit our needs, and load it into a SQLite database.
The following steps are performed in this script:
1. **Extraction**: the necessary libraries are imported and the CSV files are loaded into pandas dataframes.
2. **Initial Data Inspection**: check for missing values, duplicates, and correct data types.
3. **Cleaning & Normalization**: in this section we perform the following steps:
    -  Delete of rows that we are not going to use: Rows where *data_json* is NULL, corrections whose *id* is 5560, 5570 and 8363, corrections without *question_answers* key in *data_json* field.
    - normalize the categories to make them easier to handle.
    - eliminate the answers within the dataframe **corrections_df** corresponding to the categories *About you*, *Your studies & your work experience*, *You & Holberton*, *Your Coding Experience*, and *Housing & Financing*.
    - eliminate those questions where the *id* of the question does not correspond to any question of the **questions_df** dataframe.
    - eliminate the answers whose question is of category *unknown* and at the same time the question is of type *SelectMultiple, Input, Select* or *Scored*.
    - rename the categories of type *unknown*, whose *question_type* field is *Checkbox* and whose *evaluation_quiz_id* field value is the one indicated during the EDA.
    - remove all *Input*-type questions and their answers
    - a column with new, more general categories is generated
    - redefine the time it took the student to perform the test
    - convert the *created_at* and *updated_at* fields to UTC with detetime format in the three dataframes and also the *start_time* and *updated_at* fields of the **corrections_df** dataframe.
4. **JSON Parsing**: convert the semi-structured JSON stored in the *data_json* columns into explicit Python objects (lists of dicts/tuples).
5. **Feature Generation**: in this section we perform the following steps:
    - redefine the time it took the student to perform the test, using the criteria seen during the EDA.
    - create a binary correct flag per interaction.
    - create the *difficulty* column in the **interactions_df** dataframe using the *hit_rate* function that will return the percentage of correct answers per question, that is, the percentage of times a student answered a given question correctly.
    - eliminate the columns that we do not need from the **interactions_df** dataframe and see how it looks after all the transformations performed.
    - eliminate the columns that we do not need from the **questions_df** dataframe and see how this dataframe looks after all the transformations performed.
    - eliminate the columns that we do not need from the **quizzes_df** dataframe and see how this dataframe looks after all the transformations performed.
6. **Save Intermediate Outputs**: proceed to save them in Parquet format for later use in the Knowledge Tracing models.
7. **ETL Specification**: resume the ETL process that we have followed in this notebook.

<p align="center">
  <img src="https://i.imgur.com/F1oDkbO.png">
</p>

### [Data Splitting: Train, Validation, Test](#Data_Preprocessing/4_split.ipynb)
The data splitting is performed in the *4_split.ipynb* script. In this script, we will split the data into training, validation, and test sets.
For this file, in case it is in Colab, the file *requirements.colab.txt* must be copied to the root of the project, together with the files *interactions_clean.parquet*, *quizzes_clean.parquet* and *questions_clean.parquet*, which must be placed in the *data* folder.
The following steps are performed in this script:
1. **Imports & Data Loading**: the necessary libraries are imported and the Parquet files are loaded into pandas dataframes.
2. **Define Student‐Level Split**: split **at the user level** to prevent data leakage, that is, the model does not have access to data that it should not see during training, resulting in artificially high metrics and a model that does not generalize well to new data.
3. **Filter Interactions per Split**: create three DataFrames: **train_df**, **val_df**, **test_df**, which contain the interactions of the users assigned to each set.
4. **Verify Distributions**: check if the **distribution of interactions per-user counts** are similar across splits.
5. **Save Splits to Disk**: save the three sets of DataFrames in Parquet format for use in the scripts of the different Knowledge Tracing models.

## Knowledge Tracing Models
### [Bayesian Knowledge Tracing (BKT)](#Knowledge_Tracing_Models/1_BKT/bkt.ipynb)

In this notebook we implement, train, validate, and test a **Bayesian Knowledge Tracing** model following the paper called *Knowledge Tracing: Modeling the Acquisition of Procedural Knowledge* Corbett & Anderson (1995). This paper proposes a model to predict the probability of a student answering a question correctly based on their previous interactions with similar questions. The model is based on the idea that students learn over time and that their knowledge can be represented as a hidden state.
The following steps are performed in this script:
1. **Introduction**: this section introduces the BKT model and explains how it works.
2. **Imports & Data Loading**: we import necessary libraries and load the preprocessed data.
3. **Define BKT Class**: we define the *BayesianKnowledgeTracing* class, which is a probabilistic model used to estimate a student's mastery of a skill.
4. **Organize Data by Skill (*general_cat*) & Student**: group the data by *general_cat* and *user_id*, sort by *start_time*, and create a list of answers for each combination. We also create a dictionary to map each skill to its corresponding index in the list of skills.
5. **Grid‐Search per Skill**: search for the best **(p_L0, p_T, p_S, p_G)** that maximize the validation accuracy for each *general_cat*.
6. **Evaluate BKT on Test Set**: using the best parameters per skill, measure overall and per‐skill accuracy on the test set.
7. **Visualize Accuracy by Skill** visualize the accuracy of the model by skill, which allows us to identify patterns in the performance of the model.
8. **Example Learning Curve for a Single Skill**: example of the evolution of the probability of knowledge (p_L) across opportunities for a representative student and ability. 
9. **Conclusion**: summarize the findings and limitations of the model, as well as the steps to be taken for future improvements.

### [Performance Factors Analysis (PFA)](#Knowledge_Tracing_Models/2_PFA/pfa.ipynb)
This model is based on the paper called *Performance Factors Analysis - A New Alternative to Knowledge Tracing* by Pavlik, Cen & Koedinger (2009).
The paper proposes a new model for student learning that is based on the idea of performance factors, which are the underlying cognitive processes that influence a student's performance on a task. The model is designed to be more flexible and interpretable than traditional knowledge tracing models, and it has been shown to be effective in predicting student performance in a variety of educational contexts.
The following steps are performed in this script:
1. **Introduction**: this section introduces the BKT model and explains how it works.
2. **Imports & Data Loading**: we import necessary libraries and load the preprocessed data.
3. **Merge Skill Labels**: bring the *general_cat* (skill) into each split. Drop any interactions without a skill label.
4. **Feature Engineering**: For each interaction, compute:
    - *succ_count*: number of prior correct responses on this skill by this student.
    - *fail_count*: number of prior incorrect responses.
5. **Defining the *PFA* class**:will implement our own *PerformanceFactorAnalysis* class that:
    - **Initializes** the three parameters $\beta_{0,k},\beta_{s,k},\beta_{f,k}$.  
    - Defines the **sigmoid** helper $\sigma(x)$.  
    - Implements a **cost function** based on the *negative log‐likelihood* (equivalently binary cross‐entropy).  
    - Uses **batch gradient descent** to update $\beta_{0,k},\beta_{s,k},\beta_{f,k}$ on the training set:  
        - Compute predictions $p_i = \sigma(\beta_{0,k} + \beta_{s,k}\,S_{u,k}^{<i} + \beta_{f,k}\,F_{u,k}^{<i}\bigr)$.  
        - Compute the gradient of the loss w.r.t. each $\beta$.  
        - Update parameters: $\beta \leftarrow \beta - \eta\,\nabla_\beta$.  
    - Provides *predict_proba()* to return $\sigma(\cdot)$ and *predict(thresh=0.5)* to threshold at a chosen cutoff.
6. **Train PFA Models**: train a separate logistic regression for each skill \(k\), using features
$$
X = [\; \mathrm{succ\_count},\;\mathrm{fail\_count}\;],\quad
y = \text{correct}
$$   
7. **Validate on Held-Out Set**: we define the function called *accuracy_score*, which calculates the accuracy of a model given a set of predictions and true labels. 
8. **Test Set Evaluation**: using the trained models, now we evaluate on **test** split, again applying the corresponding **scaler**.
9. **Analysis & Visualization**: In this section the following graphs are made:
    - **PFA coefficients**: intercept vs. weights γ (events) and ρ (failures).
    - **Val vs. scatter test**: comparative accuracies.
    -  **Learning curve example**: predicted probability along the chances 4.
10. **Conclusion**: summarize the findings and limitations of the model, as well as the steps to be taken for future improvements.

### [Deep Knowledge Tracing (DKT)](#Knowledge_Tracing_Models/3_DKT/dkt.ipynb)
This model is based on the paper called *Deep Knowledge Tracing* by Piech et al. (2015). The model is a recurrent neural network (RNN) that learns to predict a student's future performance based on their past interactions with a learning system. The DKT model uses a Long Short-Term Memory (LSTM) network to capture the temporal dependencies in the student's performance data.
The following steps are performed in this script:
1. **Introduction**: this section introduces the DKT model and explains how it works.
2. **Imports & Data Loading**: we import necessary libraries and load the preprocessed data.
3. **Merge Skill Labels**: enrich the dataset by attaching the skill category (*general_cat*) associated with each question.
4. **Sequence Encoding for DKT**: need for each student a sequence of length $T$ of inputs $x_t$ and targets $y_t$:
    - Input $x_t$: one-hot vector of size $2M$ ($M$ = # distinct questions):  
        - position $q_t$ = question index if correct,  
        - position $M + q_t$ if incorrect.
    - Target $y_t$: one-hot vector of size $M$ indicating which question will be answered next, and we train only on the actual next $(q_{t+1}, a_{t+1})$.
5. **Define DKT Model**: implement the Deep Knowledge Tracing model using a single-layer **LSTM** architecture followed by a dense output layer.
    - The input shape is *(None, 2M)*, where each timestep receives a one-hot vector encoding the interaction *{question_id, correct/incorrect}*.
    - The *Masking* layer ensures that padded time steps do not affect the learning process.
    - The *LSTM* layer captures the temporal dynamics of the student's latent knowledge.
    - We apply **dropout** after the LSTM output (not recurrent dropout) to regularize the readout layer.
    - The final **Dense** layer uses a sigmoid activation to output a probability vector of length *M*, predicting the probability of correctly answering each question at the next timestep.
6. **Training Loop**: train the DKT model on the training set and evaluate its generalization using a validation set.  
Key aspects of the training loop:
    - We use a **custom data generator** to feed sequences of variable length into the model.
    - The model is trained with **early stopping**, monitoring validation AUC, and restoring the best model when no improvement is seen for 8 consecutive epochs.
    - The **AUC (Area Under the ROC Curve)** is used as the main evaluation metric, as it provides a robust measure of ranking quality for probabilistic outputs.
    - The **loss function** is binary crossentropy, computed only over meaningful prediction steps (i.e., those that have a next-question label).
7. **Validation**: visualize the evolution of the training and validation loss during training to verify model convergence and detect signs of overfitting or underfitting.
8. **Test Set Evaluation**: evaluate the model's performance on the unseen **test set** using the same data generation strategy as in training.
9. **Analysis & Visualization**: In this section the following graphs are made:
    - **Learning Curves**: visualize the predicted probability of correctly answering the *next question* at each step, compared to the actual correctness.
    - **Calibration Curve**: evaluate how well the predicted probabilities align with actual outcomes, globally.
10. **Conclusion**: summarize the findings and limitations of the model, as well as the steps to be taken for future improvements.

<p align="center">
  <img src="https://i.imgur.com/2ckefkp.png">
</p>

### [Self-Attentive Knowledge Tracing (SAKT)](#Knowledge_Tracing_Models/4_SAKT/sakt.ipynb)
This model is based on the paper called *Self-Attentive Models for Knowledge Tracing* by Pandey & Karypis (2019). The model is a self-attentive neural network that uses attention mechanisms to capture the relationships between different questions and the knowledge states of students. The model is designed to predict the probability of a student answering a question correctly based on their previous interactions with other questions.
The following steps are performed in this script:
1. **Introduction**: this section introduces the SAKT model and explains how it works.
2. **Imports & Data Loading**: we import necessary libraries and load the preprocessed data.
3. **Data Preprocessing**: prepare the raw data for training the **SAKT** model. The main tasks include:
    - **Enrichment with metadata**:
    We *merge* between user interactions and question metadata to incorporate the *general_cat* column, which represents a general category of the question. Although this column is not used directly in the SAKT model, it may be useful for further analysis or alternative models.
    - **Incomplete data filtering**:
    Interactions that have no category information (*NaN* in *general_cat*) are removed, ensuring semantic consistency in the data.
    - **Question coding**:
    A unique integer index is generated for each *question_id*, which we call *q_idx*, in the range *0* to *E-1*, where *E* is the number of unique exercises present in the training set.
    - **Interaction coding**:
    Each interaction *(question_id, correct)* is converted to a unique integer within the range *[0, 2E)* using the formula:
    $$
    y_t = q\_idx + correct \times E
    $$
    This encodes both the identity of the exercise and whether the answer was correct or not, which is key to the input of the SAKT model.
4. **Sequence Generator**: the class *SAKTSequence* is defined, which inherits from *tf.keras.utils.Sequence* and is in charge of generating the training batches efficiently.
5. **Build SAKT Model**: assemble the full Self-Attentive Knowledge Tracing (SAKT) architecture. The model processes each input sequence **X** as follows:
    - **Embeddings**  
        - **Token Embedding**: Each encoded interaction (in *[0,2E)*) is mapped to a *d_model*-dimensional vector via a learnable *Embedding(2E→d_model)*.  
        - **Positional Embedding**: A second *Embedding(max_len→d_model)* injects information about the time step, so the model is sensitive to sequence order.
    - **Add & Norm**  
    The two embeddings are summed element-wise, yielding an *(batch, max_len, d_model)* tensor.
    - **Transformer Blocks**  
    We repeat the following **n_blocks** times:
        - **Multi-Head Self-Attention** (causal)  
            Each position attends only to itself and all **previous** positions (upper triangular mask).  We project into *n_heads* subspaces of size *d_model/n_heads*, compute scaled-dot-product attention in parallel, then recombine via a final linear map.
        - **Residual + LayerNorm**  
            We add the attention output back to its input, then apply *LayerNormalization*.
        - **Position-wise Feed-Forward**  
            A two-layer MLP (*Dense(d_ff, ReLU) → Dense(d_model)*) introduces non-linearity & cross-feature mixing.
        - **Residual + LayerNorm**  
            Again add & normalize.
    - **Pooling the Last Valid Step**  
    We multiply the block output by the binary **mask** (to zero-out padding), sum over time, and divide by the total valid count.  This produces a single *d_model*-vector per student sequence.
    - **Final Prediction**  
    A *Dense(1, activation='sigmoid')* layer maps that vector to the probability *p* that the student will answer the **next** question correctly.
6. **Training Loop**: iterate through every combination, instantiate a fresh SAKT model, train it with early stopping on **validation AUC**, and record the final validation AUC & accuracy for each run.
7. **Validation**: evaluate the learning dynamics of the best SAKT model selected through hyperparameter tuning by visualizing both training and validation metrics over the epochs.
8. **Test Set Evaluation**: evaluate the generalization performance of the best SAKT model on the held-out test set, which was not used during training or validation.
9. **Analysis & Visualization**: inspect a variety of diagnostic plots to deeply understand how our SAKT model is behaving:
    - **Predicted Probability Distributions** by true class — checks separation between positives and negatives.  
    - **ROC Curve** — visualizes trade-off between true positive and false positive rates across all thresholds.  
    - **Confusion Matrix** — shows absolute counts of correct/incorrect decisions at the 0.5 threshold.  
Then we drill down to individual students (“best,” “middle,” “worst”) to see how the predicted probability evolves over time. Finally, we assess the model’s overall **calibration** by plotting actual hit‐rates against predicted probabilities.
10. **Conclusion**: summarize the findings and limitations of the model, as well as the steps to be taken for future improvements.

<p align="center">
  <img src="https://i.imgur.com/OZmxCVt.png">
</p>

### [Attentive Knowledge Tracing (AKT)](#Knowledge_Tracing_Models/5_AKT/akt.ipynb)
This model is based on the paper called *Context-Aware Attentive Knowledge Tracing* by Ghosh et al. (2020). The model is designed to predict the probability of a student answering a question correctly based on their past performance and the context of the question. Extends Self-Attentive Knowledge Tracing (SAKT) with an architecture that combines dynamic memory and attention mechanisms.
The following steps are performed in this script:
1. **Introduction**: this section introduces the AKT model and explains how it works.
2. **Imports & Data Loading**: we import necessary libraries and load the preprocessed data.
3. **Data Preprocessing**: will prepare our raw interaction records for model training by:
    - **Loading** the three splits (*train.parquet*, *val.parquet*, *test.parquet*) and question/quiz metadata.
    - **Merging** question and quiz details into each interaction, then dropping any rows missing critical fields (e.g. category).
    - **Encoding** categorical identifiers:
        - Map each *question_id* → contiguous integer in *[0, E)*.
        - Map each *quiz_id*   → contiguous integer in *[0, Qz)*.
        - Map each question’s *general_cat* → integer in *[0, C)*.
    - **Computing** two kinds of time‐since‐last features:
        - *prev_time_quiz*: elapsed seconds since the student’s previous attempt on **any** quiz, then standardized **per user** to avoid leakage.
        - *cum_attempts*: cumulative count of attempts per user (optional, may help some architectures).
    - **Casting** the *correct* flag to a binary integer and inspecting the enriched DataFrame.
4. **Sequence Generator**: build a custom *Sequence* for Keras that:
    - **Groups** interactions by *user_id* and orders by *start_time*.
    - **Slides** a window of length *max_len* over each user’s history:
        - **X**: input tensor of shape *(batch, max_len, 6)* containing
            1. *q_idx*  
            2. *quiz_idx*  
            3. *difficulty*  
            4. *prev_time_quiz*  
            5. *c_idx*  
            6. *correct_flag*  
        - **M_mask**: boolean mask *(batch, max_len)* indicating valid steps.
        - **Y**: next‐step label (the final *correct_flag*).
5. **Build AKT Model**: implement the AKT model. This model learns both **question-aware** and **response-aware** representations via two parallel Transformer-based encoders. Key components:
    - **Rasch Embeddings**: Combine per-concept embeddings and question-specific difficulty parameters (Rasch model).
    - **Monotonic Attention**: A variant of self-attention that adds time-aware constraints via distance-aware decay.
    - **Two Encoders**: One for question sequences, another for response sequences (using student correctness).
    - **Retriever Layer**: Applies cross-attention from questions to encoded responses.
    - **Prediction**: Combines the final question and retrieved response vectors to estimate *P(correct)* for the next question.
6. **Training Loop**: train our **AKT** model. The process includes:
    - Constructing the training and validation sequences using our custom *AKTSequence* class.  
    - Defining a grid of hyperparameters for tuning.  
    - Performing a **partial grid search** with early stopping on validation AUC.  
    - Logging and ranking the results by validation performance.  
    - Selecting the best configuration and retraining with full metric logging.
7. **Validation**: evaluate the learning dynamics of the best AKT model selected through hyperparameter tuning by visualizing both training and validation metrics over the epochs.
8. **Test Set Evaluation**: evaluate the generalization performance of the best AKT model on the held-out test set, which was not used during training or validation.
9. **Analysis & Visualization**: inspect a variety of diagnostic plots to deeply understand how our AKT model is behaving:
    - **Predicted Probability Distributions** by true class — checks separation between positives and negatives.  
    - **ROC Curve** — visualizes trade-off between true positive and false positive rates across all thresholds.  
    - **Confusion Matrix** — shows absolute counts of correct/incorrect decisions at the 0.5 threshold.  
Then we drill down to individual students (“best,” “middle,” “worst”) to see how the predicted probability evolves over time. Finally, we assess the model’s overall **calibration** by plotting actual hit‐rates against predicted probabilities.
10. **Conclusion**: summarize the findings and limitations of the model, as well as the steps to be taken for future improvements.

<p align="center">
  <img src="https://i.imgur.com/0F8LIVp.png">
</p>

## [Comparative Analysis Models](#Knowledge_Tracing_Models/comparative_analysis_models.ipynb)
From the analysis made in *comparative_analysis_models.ipynb* file, the following can be concluded:
    - Best Ranking Performance: DKT excels at AUC but needs calibration.
    - Best Calibration & Interpretability: SAKT offers well‑calibrated probabilities and attention‑based explanations.
    - Simplicity & Insight: BKT/PFA remain valuable for rapid prototyping and conceptual clarity.
    - Context Sensitivity: AKT innovates with IRT embeddings and time decay, but at a higher computational cost.

<p align="center">
  <img src="https://i.imgur.com/98h6yQ3.png">
</p>

Taking into account predictive performance, interpretability, computational requirements, and robustness, SAKT emerges as the most balanced choice for real‑world educational platforms:
- **Strong, Consistent Performance**: Test AUC ≈ 0.80 and accuracy ≈ 0.74 — close to the best deep models but without their training pitfalls.
- **Interpretability & Diagnostics**: Attention weights can be visualized to explain predictions and guide instructors.
- **Efficient Training & Inference**: Transformer‑based parallelism offers faster epochs than RNNs, with manageable model size (∼0.7 M params).
- **Calibrated Probabilities**: Good Brier score and calibration curve support reliable decision‑making (e.g., mastery thresholds).
- **Extensible Architecture**: Additional features (response time, content embeddings) can be integrated into the attention framework.

## [SAKT Inference Application](#SAKT_Inference_Application/inference.py)
Once we decided that SAKT was our production‐ready Knowledge Tracing model, we built a lightweight inference script (*inference.py*) to:
1. **Load** the trained model and its metadata (architecture, weights, token encodings).
2. **Stream** a student’s history of question–response interactions.
3. **Compute** both a per‐quiz mastery breakdown by category and the model’s predicted probability of success on the next question.
4. **Provide** cumulative accuracy and tailored “review recommendations” in real time.

The script loads a SAKT model and runs inference on a user. It loads the model architecture and weights from the specified paths, and the QID-to-index mapping from a pickle file. It then processes the user's interaction data, updating the history and mask as new interactions are encountered.
Finally, it predicts the probability of the next correct answer using the SAKT model, and provides general reinforcement based on the model's predictions and the user's performance.

### Example of Functionality

Once located in the *SAKT_Inference_Application* folder, the script can be executed with the following command:

```bash
$ python3 inference.py
```
After executing the script, you can see an output like the following:

```yaml
=== Report for user 1234 (269 total interactions) ===

Quiz 1:  P(next correct) = 49.5%
🎯  Cumulative accuracy: 64% (36/56)
📊  Performance by category:
    - python         : 80% (8/10)
    - c              : 50% (5/10)
    - sql            : 60% (3/5)
🧠 → Topics to review: c, sql
⚠️  Your overall readiness is low — consider revisiting fundamentals.


Quiz 2:  P(next correct) = 65.9%
🎯  Cumulative accuracy: 60.8% (79/130)
📊  Performance by category:
    - C              : 58% (43/74)
🧠 → Topics to review: C
👍  Your overall readiness is good, but consider revisiting the topics listed above.


Quiz 3:  P(next correct) = 81.0%
🎯  Cumulative accuracy: 70.3% (137/195)
📊  Performance by category:
    - Math & Stats   : 100% (13/13)
    - Web & Front-end: 100% (11/11)
    - Programming & Basics: 73% (8/11)
    - Python         : 89% (8/9)
    - Shell          : 100% (8/8)
    - Databases & ORM: 71% (5/7)
    - Javascript     : 83% (5/6)
🧠 → Topics to review: None
✅  You seem ready for the next challenge!
```

- Per‑quiz, it reports the SAKT prediction for the next question.
- Shows cumulative accuracy, helps track progress across multiple quizzes.
- Lists mastery in each general category and flags topics below the threshold (<70%).
- Delivers a concise recommendation based on both model confidence and observed performance.

### Why This Matters

- **Real‑time feedback**: Instructors (or a simple CLI) can see, immediately after each quiz, which topics a student struggles with and how confident the model is about their next attempt.
- **Balanced signals**: Combining empirical accuracy and the model’s forward‐looking probability yields richer intervention triggers than either alone.
- **Extensible & lightweight**: Since the script uses only local Python and Pandas/TensorFlow, it can be easily wrapped into a small web service (Flask/FastAPI) or scheduled batch job.

This inference application closes the loop: it turns our SAKT model from a research‐grade experiment into a practical tool for diagnostic assessment and just‐in‐time recommendations.

## Challenges, Lessons Learned & Future Directions

### Detailed Challenges

**Semi‑structured JSON parsing**
- Student responses were embedded as a JSON blob (data_json) with inconsistent keys (question_answers, items, nested lists vs. objects).
- Required robust parsing functions and multiple passes to handle missing fields, nulls, and malformed entries.

**Outlier handling in durations**
- “Time on task” had impossible values (e.g. hours-long sessions).
- We computed percentiles (5th–99th), visualized with histograms, and winsorized to cap extreme values at the 99th percentile.

**Aligning deep models with pedagogical categories**
- Transformers (SAKT/AKT) naturally treat each question as a token, ignoring skill labels.
- We experimented with including general_cat embeddings and learned post‑hoc grouping by category metrics.

**Model serialization & inference pitfalls**
- Saving a Keras model’s JSON + weights separately avoided monolithic HDF5 errors.
- Restoring the transform mask for MultiHeadAttention required rebuilding the custom architecture in code (no single load_mode).

**Balancing simplicity vs. performance**
- Baseline BKT/PFA are trivial to train and interpret, but plateaued at AUC≈0.72.
- DKT delivered AUC≈0.90 but was brittle (meaningless probability curve, calibration drift).
- SAKT struck the best balance — robust AUC≈0.79, explainable via attention, and efficient to train.

### Lessons Learned

- **Data Quality Trumps Model Complexity**
Rigorous EDA and ETL — handling missing JSON fields, normalizing categories, capping outliers — were essential. Without clean, well‑structured interaction logs, even the most advanced transformers produce unreliable predictions.

- **Balance Performance with Interpretability**
Although DKT achieved the highest raw AUC, its opaque hidden state made it hard to justify recommendations. SAKT provided nearly identical performance while exposing attention weights and per‑category mastery, enabling actionable insights.

- **Model Export & Inference Are Non‑Trivial**
Saving a Keras model as separate JSON + weights, rebuilding the architecture in code, and correctly reconstructing masks for MultiHeadAttention taught us that deployment is as challenging as training.

- **Iterative Refinement**
Early versions of the CLI demo suffered from cold‑start issues (very low probabilities with empty histories). Introducing a “burn‑in” period and combining model confidence with simple cumulative accuracy improved user trust.

- **Ethics Must Be Integrated, Not Bolted On**
From the start, we built per‑category fairness checks, privacy safeguards, and explanatory outputs — rather than treating them as afterthoughts.

### Future Directions

- Improve models: try to adjust the different models to obtain better accuracy and AUC values.
- Fix the DKT model: Try to find the problem with the DKT model, where, although the AUC value is high, the probability curve does not behave correctly.
- Multi‑Task Learning: jointly predict correctness, response time, and hint usage for richer student modeling.
- Meta‑Learning & Cold Start: adapt quickly to new students with few interactions.
- Rich Content Embeddings: integrate question text via transformer language models to capture semantic similarity.
- Deployment & A/B Testing: embed the chosen model into a production tutoring system and measure learning gains in vivo.

## Conclusion

This extensive project demonstrates that Knowledge Tracing is as much about data engineering and evaluation rigor as it is about modeling. From extracting millions of quiz logs to delivering interpretable attention maps, the journey spanned:

- **Data Pipelines**: MySQL → CSV → Parquet → Train/Val/Test.
- **Models**: Bayesian, logistic, recurrent, self‑attentive, and context‑aware.
- **Insights**: Calibration, interpretability, and context matter as much as raw ranking performance.

The choice of the “best” model ultimately depends on priorities — accuracy vs. interpretability vs. computational cost. Yet the blended approach of robust preprocessing, classical baselines, and advanced deep architectures provides a blueprint for building the next generation of adaptive learning platforms.

## Author :black_nib:
- Alberto Riffaud - [GitHub](https://github.com/alriffaud) | [Linkedin](https://www.linkedin.com/in/alberto-riffaud) | [Personal Website](https://alriffaud.github.io/)<br>
Passionate about technology, especially in the field of artificial intelligence and machine learning, I seek to contribute to the development of innovative solutions. My training and experience as a mathematics and physics teacher for 25 years have provided me with analytical skills, problem-solving and effective communication, complemented by my participation in development teams and extensive knowledge of programming languages ​​such as Python and C. I am a persevering person, responsible and oriented towards self-learning, always seeking efficiency and clarity in the code.
