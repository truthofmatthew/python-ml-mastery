# Graph Theory Basics

Graph theory is a part of mathematics that studies graphs and their properties. It's a key part of discrete mathematics, which deals with distinct or separate values. In this tutorial, we will learn about the basic concepts and terminologies in graph theory.

## Task 1: Definitions of Graphs, Vertices, Edges, and Paths

A **graph** is a set of objects where some pairs of the objects are connected by links. The interconnected objects are represented by mathematical abstractions called **vertices**, and the links that connect some pairs of vertices are called **edges**.

Here's a simple way to understand it:

Imagine a group of friends. Each friend can be seen as a vertex. The relationships between them (who knows who) can be seen as edges. This whole network of friends can be represented as a graph.

A **path** in a graph is a sequence of vertices where each adjacent pair is connected by an edge. For example, if we have vertices A, B, and C, and there are edges connecting A-B and B-C, then A-B-C is a path.

## Task 2: Special Types of Graphs

There are many special types of graphs. Here are two of them:

1. **Tree**: A tree is a graph with no cycles. A cycle is a path that starts and ends at the same vertex. In a tree, any two vertices are connected by exactly one path.

2. **Bipartite Graph**: A bipartite graph is a graph whose vertices can be divided into two disjoint sets such that every edge connects a vertex in one set to a vertex in the other set. No edge connects vertices within the same set.

## Task 3: Graph Theory Concepts - Connectivity and Cycles

**Connectivity** refers to the minimum number of elements (vertices or edges) that need to be removed for a graph to become disconnected. A graph is said to be connected if there is a path between every pair of vertices.

A **cycle** in a graph is a path that starts and ends at the same vertex and includes at least one edge. For example, if we have vertices A, B, and C, and there are edges connecting A-B, B-C, and C-A, then A-B-C-A is a cycle.

Understanding these concepts can help solve many problems in computer science, such as finding the shortest path between two points, or determining whether a network is robust against failures.

In the next lesson, we will delve deeper into graph theory and explore more complex concepts and their applications.