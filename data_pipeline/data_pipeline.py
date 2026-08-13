import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import sqlite3

BASE_URL = "https://books.toscrape.com/"

# At least 3 categories
CATEGORIES = {
    "Travel": "catalogue/category/books/travel_2/index.html",
    "Mystery": "catalogue/category/books/mystery_3/index.html",
    "Historical Fiction": "catalogue/category/books/historical-fiction_4/index.html"
}

books_data = []

for category, category_url in CATEGORIES.items():

    url = BASE_URL + category_url

    while url:

        response = requests.get(url)

        if response.status_code != 200:
            print("Failed to load:", url)
            break

        soup = BeautifulSoup(response.text, "html.parser")

        for book in soup.select("article.product_pod"):

            title = book.h3.a["title"]

            price = book.select_one(".price_color").text.strip()

            rating = book.select_one(".star-rating")["class"][1]

            availability = book.select_one(".availability").get_text(
                strip=True
            )

            books_data.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category
            })

        # Find next page
        next_button = soup.select_one("li.next a")

        if next_button:
            next_url = next_button["href"]

            if next_url.startswith("http"):
                url = next_url
            else:
                current_path = url.rsplit("/", 1)[0]
                url = current_path + "/" + next_url
        else:
            url = None

print("Total books scraped:", len(books_data))

df = pd.DataFrame(books_data)

print(df.head())
df.to_csv("books_data.csv",index=False)