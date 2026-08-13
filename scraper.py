import requests
from bs4 import BeautifulSoup
import csv

url ="https://books.toscrape.com/"
response = requests.get(url)

print("Status code:",
response.status_code)

soup = BeautifulSoup(response.text,"html.parser")

books = soup.select("article.product_pod")

print("Number of books:",len(books))

with open("books.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Price", "Rating", "Availability", "Category"])

    for book in books:
        title = book.h3.a["title"]
        price = book.select_one(".price_color").text
        rating = book.select_one("p.star-rating")["class"][1]
        availability = book.select_one(".availability").get_text(strip=True)
        link = book.h3.a["href"]

        book_url = "https://books.toscrape.com/" + link
        book_response = requests.get(book_url)
        book_soup = BeautifulSoup(book_response.text, "html.parser")

        category = book_soup.select("ul.breadcrumb li")[2].get_text(strip=True)

        writer.writerow([title, price, rating, availability, category])

        print("Title:", title)
        print("Price:", price)
        print("Rating:", rating)
        print("Availability:", availability)
        print("Category:", category)
        print() 
   
          

	

