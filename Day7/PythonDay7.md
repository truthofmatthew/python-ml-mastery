# Day 7: File Handling in Python

File handling is a crucial aspect of programming that allows us to read, write, and manage files. Python provides built-in functions for file handling, making it easy to work with files. In this tutorial, we will learn how to perform various file operations in Python.

## Task 1: Write a Python script to read a file and print its contents line by line

Python provides the `open()` function to open a file. This function returns a file object, which is used to read, write, and modify the file. Here is how you can read a file line by line:

```python
# Open the file in read mode ('r')
file = open('example.txt', 'r')

# Read the file line by line
for line in file:
    print(line)

# Close the file
file.close()
```

In the above code, we first open the file `example.txt` in read mode (`'r'`). Then, we use a for loop to read the file line by line. Finally, we close the file using the `close()` method.

## Task 2: Append a new line of text to an existing file and then read the updated file

To append text to an existing file, we can open the file in append mode (`'a'`). Here is how you can do it:

```python
# Open the file in append mode ('a')
file = open('example.txt', 'a')

# Write a new line to the file
file.write('\nThis is a new line.')

# Close the file
file.close()

# Now, let's read the updated file
file = open('example.txt', 'r')
for line in file:
    print(line)
file.close()
```

In the above code, we first open the file in append mode and write a new line to the file. Then, we close the file. After that, we open the file again in read mode and print its contents.

## Task 3: Use context managers (with statement) to manage file operations safely and efficiently

A context manager in Python handles the setup and teardown of resources. When working with files, it can automatically close the file once the operations are done, even if an error occurs. This makes the code safer and cleaner. Here is how you can use a context manager to work with files:

```python
# Open the file using a context manager
with open('example.txt', 'r') as file:
    # Read the file line by line
    for line in file:
        print(line)
```

In the above code, we use the `with` statement to open the file. This creates a context where the file is open. Inside this context, we read the file line by line. Once the context (the indented block of code) is exited, the file is automatically closed, even if an error occurs within the context.

That's it for today's lesson on file handling in Python. Remember to always close your files after you're done with them to free up system resources. Using a context manager is a good practice as it handles this automatically.