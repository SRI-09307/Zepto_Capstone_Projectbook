# Zepto Capstone Project – Book Data Analysis and Price Prediction

## Project Overview

This project collects book data, processes and cleans the data, performs analysis, stores the data in SQLite databases, predicts book prices using a machine learning model, and provides a simple support assistant to display book information.

## Project Files

```text
Zepto_Capstone_project/
│
├── analytics/
│   └── analytics.py
│
├── data_pipeline/
│   ├── data_pipeline.py
│   ├── predict_price.py
│   ├── book_price_model.pkl
│   ├── books_data.csv
│   ├── books_database.db
│   ├── books_database_normalized.db
│   ├── books_data_cleaned.csv
│   ├── books_data_encoded.csv
│   ├── books_data_final.csv
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_data_scaled.csv
│   └── test_data_scaled.csv
│
├── support_assistant/
│   └── support_assistant.py
│
├── scraper.py
├── books.csv
├── books_data.csv
└── README.md
## Technologies Used

- python
-pandas
-NumPy
-Scikit-learn
SQLite
-joblib
## How to Run
 ### 1. Run the Web Scraper
python scraper.py
The scraper collects book details such as:
Title
Price
Rating
Availability
Category
### 2. Run the Data Pipeline
python data_pipeline\data_pipeline.py
This processes the collected book data and prepares the datasets for machine learning.
3. Predict Book Price
python data_pipeline\predict_price.py
The program asks for:
Star rating
Availability
Category
Example:
Enter star rating (1-5): 4
Enter availability (In stock/Out of stock): In stock
Enter category (Mystery/Historical Fiction/Travel): Mystery

Predicted Book Price: 46.52
4. Run Analytics
cd C:\Users\Teja\OneDrive\Desktop\Zepto_Capstone_project
1. Run the Web Scraper
python scraper.py
The scraper collects book details such as:
Title
Price
Rating
Availability
Category
2. Run the Data Pipeline
python data_pipeline\data_pipeline.py
This processes the collected book data and prepares the datasets for machine learning.
3. Predict Book Price
python data_pipeline\predict_price.py
The program asks for:

- Star rating
- Availability
- Category

Example:

Enter star rating (1-5): 4
Enter availability (In stock/Out of stock): In stock
Enter category (Mystery/Historical Fiction/Travel): Mystery

Predicted Book Price: 46.52

### 4. Run Analytics

python analytics\analytics.py

The analytics program displays information such as:

- Highest priced book
- Lowest priced book
- Number of books by category
- Average price by category
- Availability information

### 5. Run Support Assistant

python support_assistant\support_assistant.py

The support assistant displays book information including:

- Title
- Price
- Star rating
- Availability
- Category

## Machine Learning

The project uses a machine learning model to predict book prices based on book-related features such as:

- Star rating
- Availability
- Category

The trained model is stored as:

data_pipeline\book_price_model.pkl

## Data Files

The project generates and uses several datasets:

- books_data.csv - collected book data
- books_data_cleaned.csv - cleaned data
- books_data_encoded.csv - encoded data
- books_data_final.csv - final processed data
- train_data.csv - training dataset
- test_data.csv - testing dataset
- train_data_scaled.csv - scaled training data
- test_data_scaled.csv - scaled testing data

## Databases

Two SQLite databases are used:

- books_database.db - SQLite database
- books_database_normalized.db - normalized SQLite database

## Analytics Results

The completed analytics process provides:

- Highest priced book
- Lowest priced book
- Category-wise book counts
- Average price by category
- Availability counts

## Project Status

The major components of the project have been implemented and tested successfully:

- Web scraping
- Data processing
- Data cleaning
- Feature encoding
- Data scaling
- Machine learning price prediction
- SQLite database creation
- Data analytics
- Support assistant

## Conclusion

This project demonstrates an end-to-end data science workflow starting from data collection and preprocessing to machine learning prediction, database storage, analytics, and a support assistant.

 
