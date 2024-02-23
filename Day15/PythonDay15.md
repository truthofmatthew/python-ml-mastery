# Day 15: Unit Testing in Python

Unit testing is a critical part of software development. It ensures that individual units of source code, such as functions or methods, work as expected. Python's `unittest` module provides tools for testing your code. In this tutorial, we will learn how to write and run unit tests using Python's `unittest` framework.

## Task 1: Write Unit Tests for a Simple Function

Let's start with a simple function that sorts a list. We will then write a unit test for this function.

```python
def sort_list(input_list):
    return sorted(input_list)
```

To write a unit test for this function, we will use the `unittest` module. A unit test checks a small component in your code. You can use it to verify the correctness of a function or method.

```python
import unittest

class TestSortList(unittest.TestCase):
    def test_sort_list(self):
        self.assertEqual(sort_list([3, 2, 1]), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
```

In the above code, `TestSortList` is a test case for the `sort_list` function. `test_sort_list` is a test method. In this method, we use the `assertEqual` method to check if the `sort_list` function returns the correct output.

## Task 2: Integrate Test Cases with Test Suite Management in Unittest

A test suite is a collection of test cases, test suites, or both. It is used to aggregate tests that should be executed together.

```python
import unittest

class TestSortList(unittest.TestCase):
    def test_sort_list(self):
        self.assertEqual(sort_list([3, 2, 1]), [1, 2, 3])

class TestSuite(unittest.TestCase):
    def test_suite(self):
        suite = unittest.TestSuite()
        suite.addTest(TestSortList('test_sort_list'))
        runner = unittest.TextTestRunner()
        runner.run(suite)

if __name__ == '__main__':
    unittest.main()
```

In the above code, `TestSuite` is a test suite that includes the `TestSortList` test case. We use the `addTest` method to add the test case to the test suite. The `TextTestRunner` class is a basic test runner that runs the test suite and outputs the results to the console.

## Task 3: Use Unittest.mock to Mock Objects and Test Functions in Isolation

The `unittest.mock` module provides a core `Mock` class. You can use it to create a mock object in your test, then specify what the return value should be when its methods are called.

```python
import unittest
from unittest.mock import Mock

def sort_list(input_list):
    return sorted(input_list)

class TestSortList(unittest.TestCase):
    def test_sort_list(self):
        mock_list = Mock()
        mock_list.return_value = [1, 2, 3]
        self.assertEqual(sort_list(mock_list()), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
```

In the above code, we create a mock object `mock_list`. We then specify that when `mock_list` is called, it should return `[1, 2, 3]`. We use this mock object as the input to the `sort_list` function and check if the function returns the correct output.

In conclusion, unit testing is a powerful tool for ensuring the correctness of your code. Python's `unittest` module provides a rich set of tools for writing and running unit tests.