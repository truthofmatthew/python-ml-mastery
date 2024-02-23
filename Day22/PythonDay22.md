# Day 22: Asynchronous Programming in Python

Asynchronous programming is a design pattern that allows the execution of operations to be paused or stopped, freeing up system resources for other tasks. This is particularly useful in I/O bound operations where waiting for data to be read or written can cause significant delays. Python's `asyncio` library provides features for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources, running network clients and servers, and other related primitives.

## Task 1: Write a Simple Coroutine with asyncio

A coroutine is a special function that can be paused and resumed, allowing it to be non-blocking. Coroutines work in the same thread, allowing concurrent execution without the need for multi-threading or multi-processing.

Here's a simple example of a coroutine using `asyncio`:

```python
import asyncio

async def hello_world():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

# Run the coroutine
asyncio.run(hello_world())
```

In this example, `hello_world` is a coroutine that prints "Hello", waits for 1 second, and then prints "World". The `await` keyword is used to pause the execution of the coroutine until the awaited operation is complete. In this case, it's waiting for `asyncio.sleep(1)` to complete.

## Task 2: Use asyncio.gather to Run Multiple Coroutines Concurrently

`asyncio.gather` is a function that runs coroutines concurrently and returns their results when all coroutines are complete.

Here's an example of using `asyncio.gather`:

```python
import asyncio

async def count_to_three():
    for i in range(1, 4):
        print(i)
        await asyncio.sleep(1)

async def count_down_from_three():
    for i in range(3, 0, -1):
        print(i)
        await asyncio.sleep(1)

# Run the coroutines concurrently
asyncio.run(asyncio.gather(count_to_three(), count_down_from_three()))
```

In this example, `count_to_three` and `count_down_from_three` are coroutines that count to three and count down from three, respectively, with a 1-second delay between each print. `asyncio.gather` is used to run these coroutines concurrently.

## Task 3: Fetch Data from Multiple URLs Concurrently Using aiohttp

`aiohttp` is a library for making HTTP requests in an asynchronous way. It can be used with `asyncio` to fetch data from multiple URLs concurrently.

Here's an example of using `aiohttp` to fetch data from multiple URLs:

```python
import aiohttp
import asyncio

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    urls = ['http://python.org', 'http://google.com']
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Run the main coroutine
results = asyncio.run(main())
print(results)
```

In this example, `fetch` is a coroutine that fetches data from a URL using an `aiohttp.ClientSession`. `main` is a coroutine that creates a `ClientSession` and uses it to fetch data from multiple URLs concurrently using `asyncio.gather`. The results are then printed out.