# OpenAI's tokenizer
!pip install tiktoken -q

import tiktoken
import numpy as np

# predefined vocabulary of tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

print(f"Vocabular size: {tokenizer.n_vocab:,} tokens")
# n_vocab: unique tokens in the encoding can present

"""## Basic Tokenziation

text -> numbers
"""

text = "Hello World"
tokens = tokenizer.encode(text)

print(f"Text:'{text}'")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Representation of eahc token
print("\nToken breakdown: ")
for token in tokens:
  print(f" {token} -> '{tokenizer.decode([token])}'")

# Hello: one token
# ' space' is another token

# other examples
examples = [
    "Hello world",
    "don't",
    "artificial intelligence",
    "I love AI",
    "supercalifragilisticexpialidocious",
    "🚀",
    "café",
    "    spaces    "

]

print("How different texts get tokenized:\n")
for text in examples:
  tokens = tokenizer.encode(text)
  print(f"'{text}")
  print(f" -> {len(tokens)} tokens: {tokens}")

  # pieces shows exactly how the tokenizer split the original text into token-level text segments
  pieces = [tokenizer.decode([t]) for t in tokens]
  print(f" -> pieces: {pieces}")
  print()

my_text = 'my name is ABD'

tokens = tokenizer.encode(my_text)
print(f"Text: '{my_text}")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")
print(f"Pieces: {[tokenizer.decode([t]) for t in tokens]}")

"""Token count matters:
- API costs that charge per token
- Context limits
- Model behavior; some tasks break across token boundaries
"""

test_cases = {
    "English prose": "The quick brown fox jumps over the lazy dog.",
    "Python code": "def hello():\n    print('Hello, world!')",
    "JSON": '{"name": "Alice", "age": 30, "city": "NYC"}',
    "Numbers": "1234567890 9876543210 1111111111",
    "URL": "https://www.example.com/path/to/page?query=value"
}

print("Token efficiency comparison:\n")
for name, text in test_cases.items():
  tokens = tokenizer.encode(text)
  chars = len(text)
  ratio = chars / len(tokens)
  print(f"{name}")
  print(f" {chars} chars -> {len(tokens)} tokens ({ratio:.1f} chars/token)")
  print()

"""**Key Insights**

Different content has different token efficiency

English Prose: ~4 chars per token

Code/JSON: often less efficient

Affects API costs

### Token Boundary Problem
Some tasks are hard because they require reasoning WITHIN tokens.
"""

# LLMs struggle with letter counting
word = 'strawberry'
tokens = tokenizer.encode(word)

print(f"Word: '{word}")
print(f"Tokens: {tokens}")
print(f"Pieces: {[tokenizer.decode([t]) for t in tokens]}")
print()
print("The model sees these as pieces, not individual letters")
print("Counting 'r's requires looking Inside tokens which is hard")

# reversing words

word = "hello"
tokens = tokenizer.encode(word)
print(f"'{word}' -> tokens: {[tokenizer.decode([t]) for t in tokens]}")
print()
print("If 'hello' is ONE token, the model can't easily reverse it.")
print("It would need to decompose something it sees as atomic")

"""## Embeddings

An embedding converts a token ID into a vector of numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Simulating an embedding matrix
vocab_size = 50000
embedding_dim = 768

embedding_matrix = np.random.rand(vocab_size, embedding_dim) * 0.02

print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f" -> {vocab_size:,} tokens")
print(f" -> {embedding_dim} dimensions per token")
print(f" -> {vocab_size * embedding_dim:,} total parameters just for embeddings")

# How embedding lookup works

text = "Hello World"
tokens = tokenizer.encode(text)

print(f"Text: '{text}'")
print(f"Token IDs: {tokens}")
print()

for token_id in tokens:
  embedding = embedding_matrix[token_id]
  print(f"Token {token_id} ('{tokenizer.decode([token_id])}')")
  print(f" -> Embedding Shape: {embedding.shape}")
  print(f" -> First 10 values: {embedding[:10].round(3)}")
  print()

"""# Real Embeddings: Word Similarity"""

!pip install gensim -q

import gensim.downloader as api

# Load pre-trained word vectors
print("Loading word vectors...")
word_vectors = api.load("glove-wiki-gigaword-100")
print(f"Loaded! Vocabulary: {len(word_vectors):,} words")

# Finding similar words

word = 'king'
similar = word_vectors.most_similar(word, topn = 10)

print(f"Words most similar to '{word}':\n")
for similar_word, score in similar:
  print(f" {similar_word}: {score:.3f}")

my_word = 'computer'

if my_word in word_vectors:
  similar = word_vectors.most_similar(my_word, topn=10)
  print(f"Words most similar to '{my_word}':\n")
  for w, score in similar:
    print(f"{w}: {score:.3f}")
else:
  print(f"'{my_word}' not in vocabulary. Try another word.")

result = word_vectors.most_similar(
    positive = ["king", "woman"],
    negative = ["man"],
    topn = 5
)

print("king - man + woman = ?\n")
for word, score in result:
  print(f" {word}:{score:.3f}")

analogies = [
    (["paris", "germany"], ["france"], "paris - france + germany = ?"),
    (["bigger", "cold"], ["big"], "bigger - big + cold = ?"),
]

for positive, negative, description in analogies:
  print(description)
  result = word_vectors.most_similar(positive=positive, negative=negative, topn=3)
  for word, score in result:
    print(f" {word}: {score:.3f}")
  print()

word_groups = {
    "royalty": ["king", "queen", "prince", "princess", "royal", "throne"],
    "family": ["man", "woman", "boy", "girl", "father", "mother"],
    "animals": ["dog", "cat", "horse", "bird", "fish", "lion"],
    "tech": ["computer", "software", "internet", "digital", "data", "code"],
}

words = []
embeddings = []
colors = []
color_map = {"royalty": "purple", "family": "blue", "animals": "green", "tech": "red"}

for group, word_list in word_groups.items():
  for word in word_list:
    if word in word_vectors:
      words.append(word)
      embeddings.append(word_vectors[word])
      colors.append(color_map[group])

embeddings = np.array(embeddings)
print(f"Collected {len(words)} words, each with {embeddings.shape[1]} dimensions")

# Reduce to 2d using PCA
# for plotting purposes
pca = PCA(n_components=2)
# many dimensions to 2 dimensions
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
for i, word in enumerate(words):
# for each word and its embedding
  plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c = colors[i], s=100)
  # plt.scatter(x, y, color of point, size of dot)
  plt.annotate(word, (embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1] + 0.1))
  # writing the word next to its point, 0.1 shifts it slightly to avoid overlap

plt.title("Word Embeddings Projected to 2D")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=g) for g, c in color_map.items()]
plt.legend(handles=legend_elements)
plt.grid(True, alpha=0.3)
plt.show()

"""Words Cluster by meaning

- Royaty words near each other

- Family words form a cluster

- This structure emerges from training

# Positional Encoding
*Injecting word order*
"""

# same word, different order, different meaning

sentence1 = "dog bites man"
sentence2 = "man bites dog"

token1 = tokenizer.encode(sentence1)
token2 = tokenizer.encode(sentence2)

print(f"'{sentence1}' -> {token1}")
print(f"'{sentence2}' -> {token2}")
print()
print("Same tokens, reordered")
print("Without position info, model can't tell these apart")

# As transformers dont read text seq, they need extra info about position
def get_positional_encoding(seq_length, d_model):
# (number of positions, embedding size)
  '''
  Generate sinusoidal positional encodings.

  Args:
      seq_length: Maximum sequence length
      d_model: Embedding dimension

  Returns:
      Positionl encoding of matrix of shape(seq_length, d_model)
  '''
  position = np.arange(seq_length)[:, np.newaxis]
  # creates a row representing position index with shape (100, 1)
  div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
  # np.arange(0, d_model, 2): gives [0, 2, 4, 6...62]
  # each index is scaled: -(log(10000) / d_model): scaling factor

  pe = np.zeros((seq_length, d_model))
  pe[:, 0::2] = np.sin(position * div_term)
  pe[:, 1::2] = np.cos(position * div_term)

  return pe

seq_length = 100
d_model = 64

pe = get_positional_encoding(seq_length, d_model)
print(f"Postional encoding shape: {pe.shape}")
print(f" -> {seq_length} positions")
print(f" -> {d_model} dimensions")

plt.figure(figsize=(14, 6))
plt.imshow(pe.T, aspect='auto', cmap='RdBu')
plt.colorbar(label='Value')
plt.xlabel('Position in sequence')
plt.ylabel('Encoding dimension')
plt.title('Positional Encoding: Each position has a unique pattern')
plt.show()

positions_to_show = [0, 1, 2, 10, 50]

plt.figure(figsize = (12, 4))
for pos in positions_to_show:
  plt.plot(pe[pos, :20], label=f'Position {pos}', alpha=0.7)

plt.xlabel('Dimension')
plt.ylabel('Encoding Value')
plt.title('Positional Encodings for Different Positions (first 20 dims)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ADDING position encoding to embedding

text = "The cat sat"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# Get embeddings
d_model = 768
embeddings_matrix = np.random.randn(50000, d_model) * 0.02
embeddings = np.array([embeddings_matrix[t] for t in tokens])
print(f"\nEmbeddings shape: {embeddings.shape}")

# Getting PE
pe = get_positional_encoding(len(tokens), d_model)
print(f"Positional encodings shape: {pe.shape}")  # (3, 768)

# ADD them
final_embeddings = embedding + pe
print(f"Final embeddings shape: {final_embeddings.shape}")  # (3, 768)

print("\nNow each token's representation includes BOTH meaning AND position!")

the_pos0 = embedding_matrix[tokenizer.encode("the")[0]] + get_positional_encoding(1, d_model)[0]
the_pos5 = embedding_matrix[tokenizer.encode("the")[0]] + get_positional_encoding(6, d_model)[5]
difference = np.linalg.norm(the_pos0 - the_pos0)
print(f"Same word 'the' at position 0 vs position 5:")
print(f"  Euclidean distance: {difference:.4f}")
print(f"\nThe same word has DIFFERENT representations at different positions!")
