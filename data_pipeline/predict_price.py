import joblib
import pandas as pd

# Load trained model
model = joblib.load("book_price_model.pkl")

# Get user inputs
star_rating = float(input("Enter star rating (1-5): "))
availability = input("Enter availability (In stock/Out of stock): ")
category = input("Enter category (Mystery/Historical Fiction/Travel): ")

# Convert availability to number
availability_num = 1 if availability.lower() == "in stock" else 0

# Create category columns
category_Historical_Fiction = 1 if category.lower() == "historical fiction" else 0
category_Mystery = 1 if category.lower() == "mystery" else 0
category_Travel = 1 if category.lower() == "travel" else 0

# Create input data
input_data = pd.DataFrame({
    "star_rating": [star_rating],
    "availability_num": [availability_num],
    "category_Historical Fiction": [category_Historical_Fiction],
    "category_Mystery": [category_Mystery],
    "category_Travel": [category_Travel]
})

# Predict price
predicted_price = model.predict(input_data)

print("Predicted Book Price:", round(predicted_price[0], 2))