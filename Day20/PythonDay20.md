# Day 20: Introduction to APIs with Python

APIs, or Application Programming Interfaces, are a set of rules that allow different software applications to communicate with each other. They define the methods and data formats that a program can use to communicate with the API. In this tutorial, we will learn how to interact with web APIs to fetch and send data programmatically using Python.

## Learning Objective

By the end of this tutorial, you will be able to use Python to request data from and send data to APIs.

## Task 1: Use the requests library to make a GET request to a public API and print the results

Python's `requests` library is a simple yet powerful HTTP library. It allows you to send HTTP requests using Python. The HTTP request returns a Response Object with all the response data (content, encoding, status, etc.).

Let's make a GET request to a public API. For this example, we will use the JSONPlaceholder API.

```python
import requests

response = requests.get('https://jsonplaceholder.typicode.com/posts')

print(response.text)
```

In the above code, we are sending a GET request to the URL 'https://jsonplaceholder.typicode.com/posts'. The response of this request is stored in the `response` variable. We then print the text of the response using `response.text`.

## Task 2: Parse JSON received from an API call and extract specific information

The data received from an API is often in JSON format. Python has a built-in package called `json` to work with JSON data.

Let's parse the JSON data we received in the previous task and extract specific information.

```python
import requests
import json

response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = json.loads(response.text)

for post in data:
    print(post['title'])
```

In the above code, we are parsing the JSON data using `json.loads()` and storing it in the `data` variable. We then iterate over each post in the data and print its title.

## Task 3: Make a POST request to an API to send data and analyze the response

A POST request is used to send data to a server to create/update a resource. The data sent to the server with POST is stored in the request body of the HTTP request.

Let's make a POST request to the JSONPlaceholder API.

```python
import requests
import json

data = {'title': 'foo', 'body': 'bar', 'userId': 1}
response = requests.post('https://jsonplaceholder.typicode.com/posts', data=json.dumps(data))

print(response.text)
```

In the above code, we are sending a POST request to the URL 'https://jsonplaceholder.typicode.com/posts'. We are sending the data {'title': 'foo', 'body': 'bar', 'userId': 1} in the body of the POST request. The response of this request is stored in the `response` variable. We then print the text of the response using `response.text`.

In this tutorial, we learned how to interact with web APIs using Python. We learned how to make GET and POST requests and how to parse JSON data. This knowledge is fundamental when working with web services and building applications that consume APIs.