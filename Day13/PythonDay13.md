# Day 13: Concurrency in Python: Threads and Processes

Concurrency in Python is a complex subject that can significantly improve the performance of your programs. It allows multiple parts of a program to execute simultaneously, which can be particularly useful in I/O-bound and CPU-bound programs. In this tutorial, we will explore two main forms of concurrency in Python: threading and multiprocessing.

## Task 1: Create a multi-threaded application that performs several downloads concurrently

Threading is a technique for decoupling tasks which are not sequentially dependent. Threads are lighter than processes, and share the same memory space. This is particularly useful for I/O-bound tasks, such as downloading files from the internet.

Here is an example of a multi-threaded application that performs several downloads concurrently:

```python
import threading
import urllib.request

def download(url, filename):
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename}")

urls = ["http://example.com/1.jpg", "http://example.com/2.jpg", "http://example.com/3.jpg"]
filenames = ["1.jpg", "2.jpg", "3.jpg"]

for url, filename in zip(urls, filenames):
    threading.Thread(target=download, args=(url, filename)).start()
```

In this example, we create a new thread for each download. The `download` function is the target of each thread, and the `url` and `filename` are passed as arguments to this function.

## Task 2: Write a multiprocessing script to perform CPU-intensive computations in parallel

Multiprocessing is another form of concurrency that uses separate processes instead of threads. Each process has its own memory space, which makes it a good choice for CPU-bound tasks that require a lot of computations.

Here is an example of a multiprocessing script that performs a CPU-intensive computation in parallel:

```python
import multiprocessing

def compute(n):
    result = sum(i*i for i in range(n))
    print(f"Computed {result}")

numbers = [10000000, 20000000, 30000000]

if __name__ == "__main__":
    processes = [multiprocessing.Process(target=compute, args=(n,)) for n in numbers]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
```

In this example, we create a new process for each computation. The `compute` function is the target of each process, and the `n` is passed as an argument to this function.

## Task 3: Compare the performance and use cases of multithreading vs. multiprocessing

In Python, due to the Global Interpreter Lock (GIL), multiple threads cannot execute Python bytecodes simultaneously on multiple cores. Therefore, multithreading is best suited for I/O-bound tasks where the program spends most of its time waiting for external resources.

On the other hand, multiprocessing can take full advantage of multiple cores because each process runs in its own Python interpreter with its own GIL. Therefore, multiprocessing is best suited for CPU-bound tasks where the program spends most of its time doing computations.

In terms of performance, multiprocessing generally provides a significant speedup for CPU-bound tasks, while multithreading can improve the responsiveness and throughput of I/O-bound tasks.

In conclusion, the choice between multithreading and multiprocessing depends on the nature of the tasks your program needs to perform. For I/O-bound tasks, multithreading is usually a better choice, while for CPU-bound tasks, multiprocessing is usually more efficient.