import pandas as pd

# Load the final processed book data
file_path = "data_pipeline/books_data_final.csv"
df = pd.read_csv(file_path)

print("\n===== BOOK ANALYTICS =====")

# Total books
print("\nTotal books:", len(df))

# Average price
print("Average price:", round(df["price"].mean(), 2))

# Average rating
print("Average rating:", round(df["star_rating"].mean(), 2))

# Highest price
highest = df.loc[df["price"].idxmax()]
print("\nHighest priced book:")
print("Title:", highest["title"])
print("Price:", highest["price"])

# Lowest price
lowest = df.loc[df["price"].idxmin()]
print("\nLowest priced book:")
print("Title:", lowest["title"])
print("Price:", lowest["price"])

# Books by category
print("\nBooks by category:")
print(df["category"].value_counts())

# Average price by category
print("\nAverage price by category:")
print(df.groupby("category")["price"].mean().round(2))

# Availability summary
print("\nAvailability:")
print(df["availability"].value_counts())

print("\n===== ANALYTICS COMPLETED =====")