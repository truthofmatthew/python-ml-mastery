# Day 27: Python for Cybersecurity

Python is a versatile language that is widely used in the field of cybersecurity. Its simplicity and extensive library support make it an ideal choice for scripting and automating security-related tasks. In this tutorial, we will explore how to use Python for basic cybersecurity tasks such as network scanning, information gathering, and hash cracking.

## Task 1: Network Scanning with Python

Network scanning is a process used in cybersecurity to identify active hosts (computers and servers) and the services they offer. This information can be used to identify potential vulnerabilities. We will write a Python script that scans a network for open ports using the `socket` module.

```python
import socket

def scan_ports(ip, start_port, end_port):
    for port in range(start_port, end_port+1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((ip, port))
        if result == 0:
            print(f"Port {port} is open")
        sock.close()

scan_ports('127.0.0.1', 1, 1024)
```

This script attempts to establish a TCP connection to each port in the specified range. If the connection is successful, it means the port is open.

## Task 2: Information Gathering with Python

Information gathering is another crucial aspect of cybersecurity. It involves collecting detailed information about a target, such as an IP address or domain. We can automate this process using Python. In this task, we will use the `socket` and `requests` modules to gather information about a given IP address or domain.

```python
import socket
import requests

def get_info(domain):
    ip = socket.gethostbyname(domain)
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    print(response.json())

get_info('google.com')
```

This script first resolves the domain name to an IP address using the `gethostbyname` function. It then sends a GET request to the ipinfo.io API, which returns information about the IP address in JSON format.

## Task 3: Hash Cracking with Python

Hash functions are widely used in cybersecurity for storing passwords and other sensitive data. A hash function takes an input and returns a fixed-size string of bytes. The output is unique to each unique input. Hash cracking is the process of finding the original input given its hash output. In this task, we will implement a simple hash cracker in Python using a dictionary attack.

```python
import hashlib

def crack_hash(hash_value, dictionary_file):
    with open(dictionary_file, 'r') as file:
        for line in file:
            word = line.strip()
            if hashlib.md5(word.encode()).hexdigest() == hash_value:
                print(f"Found plaintext: {word}")
                return
    print("Plaintext not found in dictionary")

crack_hash('5f4dcc3b5aa765d61d8327deb882cf99', 'dictionary.txt')
```

This script reads each word from the dictionary file, computes its MD5 hash, and compares it to the given hash value. If a match is found, it prints the word and returns.

Please note that these scripts are for educational purposes only. Misuse of these scripts can lead to legal consequences. Always obtain proper authorization before performing any kind of network scanning or penetration testing.