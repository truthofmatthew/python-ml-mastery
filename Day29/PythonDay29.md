# Python and Blockchain: A Simple Implementation

Blockchain technology has revolutionized the way we think about data integrity and decentralization. In this tutorial, we will explore the basic concepts of blockchain and implement a simple version using Python.

## Task 1: Define a Blockchain Structure

A blockchain is a chain of blocks, where each block contains data. The data can be transactions, contracts, or any digital information. Each block is linked to the previous block through a cryptographic hash. Let's start by defining the basic building block—'block'.

```python
import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash
```

In the above code, we define a `Block` class with five attributes: `index`, `previous_hash`, `timestamp`, `data`, and `hash`. The `hash` attribute is a cryptographic hash of the block's index, previous block's hash, timestamp, and data.

## Task 2: Add Blocks to the Chain

Next, we need to add functionality to append new blocks to the chain. But before we do that, we need to create the genesis block—the first block in the blockchain.

```python
def create_genesis_block():
    return Block(0, "0", int(time.time()), "Genesis Block", calculate_hash(0, "0", int(time.time()), "Genesis Block"))

def calculate_hash(index, previous_hash, timestamp, data):
    value = str(index) + str(previous_hash) + str(timestamp) + str(data)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def create_new_block(previous_block, data):
    index = previous_block.index + 1
    timestamp = int(time.time())
    hash = calculate_hash(index, previous_block.hash, timestamp, data)
    return Block(index, previous_block.hash, timestamp, data, hash)
```

In the above code, `create_genesis_block` function creates the first block. The `calculate_hash` function calculates the hash of a block. The `create_new_block` function creates a new block with the given data and adds it to the blockchain.

## Task 3: Implement Proof-of-Work

To secure the blockchain against tampering, we implement a basic proof-of-work mechanism. The idea is to make it computationally difficult to add new blocks, so that it's not feasible to alter the blockchain.

```python
def proof_of_work(last_proof):
    proof = 0
    while not valid_proof(last_proof, proof):
        proof += 1
    return proof

def valid_proof(last_proof, proof):
    guess = f'{last_proof}{proof}'.encode()
    guess_hash = hashlib.sha256(guess).hexdigest()
    return guess_hash[:4] == "0000"
```

In the above code, `proof_of_work` function finds a number that when hashed with the previous proof, the hash starts with four zeroes. The `valid_proof` function checks if the hash of a proof starts with four zeroes.

This is a simple implementation of a blockchain in Python. It's not a full-fledged blockchain, but it gives you a basic understanding of how blockchain works.