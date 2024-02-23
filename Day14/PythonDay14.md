# Day 14: Web Scraping with Python

Web scraping is a method used to extract large amounts of data from websites. The data on the websites are unstructured, and web scraping enables us to convert that data into a structured form. Python is widely used for web scraping due to its rich ecosystem of libraries, including Beautiful Soup, Scrapy, and Selenium.

## Task 1: Install and use Beautiful Soup to parse HTML content from a webpage

Beautiful Soup is a Python library that is used for web scraping purposes to pull the data out of HTML and XML files. It creates a parse tree from page source code that can be used to extract data in a hierarchical and more readable manner.

To install Beautiful Soup, use pip:

```python
pip install beautifulsoup4
```

Here is a simple example of how to use Beautiful Soup to parse HTML content:

```python
from bs4 import BeautifulSoup
import requests

# Make a request to the website
r = requests.get("http://www.example.com")
r.content

# Use the 'html.parser' to parse the page
soup = BeautifulSoup(r.content, 'html.parser')

# Print the parsed data of html
print(soup.prettify())
```

## Task 2: Extract and print all the URLs found within a webpage

Beautiful Soup can be used to extract all the URLs found within an anchor tag (`<a>`) in a webpage:

```python
from bs4 import BeautifulSoup
import requests

# Make a request
r = requests.get("http://www.example.com")
soup = BeautifulSoup(r.content, 'html.parser')

# Find and print all URLs
for link in soup.find_all('a'):
    print(link.get('href'))
```

## Task 3: Implement error handling to manage non-responsive URLs or missing elements

When scraping websites, it's important to implement error handling to manage non-responsive URLs or missing elements. This can be done using Python's built-in exception handling features:

```python
from bs4 import BeautifulSoup
import requests

# Make a request
try:
    r = requests.get("http://www.example.com", timeout=5)
    r.raise_for_status()  # Raise an exception if the request was unsuccessful
except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as err:
    print(f"Error occurred: {err}")
else:
    soup = BeautifulSoup(r.content, 'html.parser')

    # Try to find and print a specific element
    try:
        print(soup.find('non_existent_tag').prettify())
    except AttributeError:
        print("Tag was not found on the page")
```

In this example, we first try to make a request to the website. If the request is unsuccessful (e.g., the website is down or the request times out), an exception is raised and the error is printed. If the request is successful, we then try to find and print a specific element on the page. If the element does not exist, an `AttributeError` is raised and a message is printed.