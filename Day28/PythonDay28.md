# Day 28: Natural Language Processing (NLP) with Python

Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans through the natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human language in a valuable way.

Python provides several libraries for NLP, including NLTK (Natural Language Toolkit) and spaCy. These libraries provide easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

## Task 1: Use NLTK to Tokenize Text into Words and Sentences

Tokenization is the process of breaking down text into words, phrases, symbols, or other meaningful elements called tokens. The input to the tokenizer is a Unicode text, and the output is a Doc object.

```python
import nltk
nltk.download('punkt')

text = "This is a sentence. So is this one."
sentences = nltk.sent_tokenize(text)
print(sentences)
# Output: ['This is a sentence.', 'So is this one.']

words = nltk.word_tokenize(text)
print(words)
# Output: ['This', 'is', 'a', 'sentence', '.', 'So', 'is', 'this', 'one', '.']
```

## Task 2: Perform Part-of-Speech Tagging and Named Entity Recognition Using spaCy

Part-of-speech tagging is the process of marking up a word in a text as corresponding to a particular part of speech, based on both its definition and its context. Named Entity Recognition (NER) is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.

```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
# Output: Apple Apple PROPN NNP nsubj Xxxxx True False
# ...

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
# Output: Apple 0 5 ORG
# ...
```

## Task 3: Explore Word Vector Representations by Generating and Analyzing Word Embeddings with spaCy

Word embeddings are a type of word representation that allows words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging natural language processing problems.

```python
tokens = nlp("dog cat banana")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
# Output: dog dog 1.0
# ...
```

In this tutorial, we have covered some basic NLP tasks using Python's NLTK and spaCy libraries. There are many more advanced tasks and techniques in NLP, such as sentiment analysis, text classification, and information extraction, which you can explore further.