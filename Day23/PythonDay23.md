# Python and SQL for Database Management

Python, with its simplicity and vast library support, is a great language for interacting with SQL databases. This tutorial will guide you through the process of performing CRUD (Create, Read, Update, Delete) operations on an SQL database using Python.

## Task 1: Establish a Connection to a SQLite Database Using sqlite3

SQLite is a self-contained, serverless, and zero-configuration database engine. Python's `sqlite3` module allows us to interact with SQLite databases.

```python
import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('my_database.db')

# Create a cursor object
cursor = conn.cursor()
```

The `connect()` function establishes a connection to the SQLite database. If the database does not exist, it will be created. The `cursor()` function returns a cursor object, which is used to execute SQL commands.

## Task 2: Create Tables and Insert Data into Them Using SQL Commands Executed from Python

We can use the `execute()` method of the cursor object to run SQL commands.

```python
# Create a table
cursor.execute("""
    CREATE TABLE employees(
        id INTEGER PRIMARY KEY,
        name TEXT,
        position TEXT,
        hire_date TEXT
    )
""")

# Insert data into the table
cursor.execute("""
    INSERT INTO employees VALUES(
        1,
        'John Doe',
        'Software Engineer',
        '2020-01-01'
    )
""")

# Commit the transaction
conn.commit()
```

The `execute()` method runs the SQL command passed as a string. The `commit()` method saves the changes made during the current transaction.

## Task 3: Query the Database for Specific Information and Update Records from a Python Script

We can retrieve data from the database using the `SELECT` command and update it using the `UPDATE` command.

```python
# Query the database
cursor.execute("SELECT * FROM employees WHERE position = 'Software Engineer'")
print(cursor.fetchall())

# Update a record
cursor.execute("""
    UPDATE employees
    SET position = 'Senior Software Engineer'
    WHERE name = 'John Doe'
""")

# Commit the transaction
conn.commit()
```

The `fetchall()` method retrieves all rows returned by the `SELECT` command. The `UPDATE` command modifies the specified records in the table.

Remember to always close the connection when you're done interacting with the database.

```python
# Close the connection
conn.close()
```

This tutorial has shown you how to perform basic database operations using Python and SQLite. With these skills, you can now manage SQL databases effectively using Python.