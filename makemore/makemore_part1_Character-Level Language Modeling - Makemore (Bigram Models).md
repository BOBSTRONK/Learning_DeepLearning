## Part 1 (Statistical Approach)
Project “**Makemore**” is a repository intended to build out character-level language models. Makemore takes a dataset and learns to generate “more” of that data.
The dataset we use here is name.txt, which contains approximately 32,000 unique names. 
- **Objective**: The model will train on these names to generate new, unique sequences that “sound” name-ike but aren’t necessarily in the original list.  
- The Model treats each name as sequence of individual characters. It learns to **predict the next character** in a sequence **given the previous one**.
In this chapter, we introduces the **Bigram language model**, the simplest type of language model to start with. A Bigram model always **works with two characters** at time. It **looks at one character and tries to predict the next one in the sequence**.
We can use **zip** Trick to iterate over a word and the same word’s next position (offset by one character).  The function **zip()** takes two (or more) separate iterable,  it takes the item at index 0 from list A and index 0 from list B and puts them together. Then it moves to index 1, and so on. It returns an **iterator of tuples**, in your loop, you unpack these these tuples immediately into ch1 and ch2. . For example:
```python
words = open('names.txt', 'r').read().splitlines()
# word[0] is emma

for w in words[:1]:
    # zip() stops iterating as soon as the shortest input runs out of items,
    # since w[1:] ("mma") is only 3 characters long, the llop stops after 3 pairs, ignoring the final "a" in the frist string.
    for ch1,ch2 in zip(w,w[1:]):

        print(ch1,ch2)
#e m
#m m 
#m a
```
Karpathy begins  by implementing a simple method to learn the statistics of the names using a python dictionary:
He iterates through the bigrams (character pairs) in the dataset and uses a dictionary to **count the frequency of each pair**. He sorts the dictionary to reveal the most common bigrams. For example, the character `n` is a very common ending character, and `a` is often followed by `n`. While effective, using a dictionary is not the most efficient way to represent or manipulate this data for machine learning purposes.
```python
b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    # b.get(key,default)
    # key: the item you are looing for
    # default: what to return if the key is not found.
    # Scenario A: if the bigram doesn't exist, the bigram is added to
    # dictionary with a count of 1
    # Sceanrio B: Bigram exists, ignores the default and return the number
    # will also do the calculation 5 + 1 = 6. Reult: The count is updated to 6.
    b[bigram] = b.get(bigram, 0) + 1
```
To handle the data more efficiently, the Pytorch is introduced. and its fundamental building block: The Tensor.  a 2-dimensional array (matrix) is a better way to store bigram counts, where **rows** represent the **first character** and **columns** represent the **second**. Tensors allow for efficient mathematical operations on these arrays.  
He demonstrates how to import PyTorch (`import torch`) and create a tensor filled with zeros using `torch.zeros()`. He also touches on data types, specifying `dtype=torch.int32` since they are counting integers.
```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
```
He creates a **28x28 matrix** (initially) to account for the 26 letters of the alphabet plus special start/end tokens.  To use characters as indices for the matrix, he builds a **lookup table** (`stoi`) that maps every character to a specific integer (e.g., 'a' -> 0, 'b' -> 1).  He rewrites the counting loop to update the tensor instead of the dictionary. For every bigram `(ch1, ch2)`, he increments the value at `N[stoi[ch1], stoi[ch2]]`.
```python
# retrieve any individual character and sort it

chars = sorted(list(set(''.join(words))))
# enumerate takes a iterable and returns a List of tuple with (index,individual_item)
# this will return a dictionary with key = value and value = index like {'a' : 1}

stoi = {s:i+1 for i,s in enumerate(chars)}

stoi['.'] = 0

# reverse value with key
itos = {i:s for s,i in stoi.items()}
```
Once the matrix is populated, Karpathy uses **Matplotlib** to visualize the counts.
```python
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(16,16))

"cmap=Blues means higher values will be darker blue"

plt.imshow(N, cmap='Blues')

for i in range(27):

    for j in range(27):

        # Construct the bigram string for this cell

        # itos[i] is the first character (Row), itos[j] is the second (column)

        chstr = itos[i] + itos[j]

        # plt.text(x,y,s,...): x,y determine where two arguments goes

        # s is the string

        # place the character string (e.g., "em") in the center of the cell

        # slightly shifted to the bottom (va=bottom)

        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')

        # Place the raw count (N[i,j]) in the center of the cell

        # slightly shifted to the top (va="top"), so it sits below the letters

        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')

plt.axis('off');
```
![[Pasted image 20261009135201.png]]How to turn the raw counts collected in the matrix into probabilities that can be used for sampling? Karpathy selected the **first row** of the **count matrix** (which represents how likely each character is to start a word). He converts these integer counts to floating-point numbers and divides them by their sum. This creates a probability vector `p` where all values sum to 1.0. For example, if ‘***m***’ starts with many words in the dataset, its corresponding value in this vector will be high.
![[Pasted image 20261009135430.png]]
- 4410 = number of times one name starts with "a"
```python
p = N[0].float()
# probability of the character begin the first character in the name
p = p / p.sum()
p
# tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273, 0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029, 0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290])
```
To actually generate data, the model needs to “roll the dice” based on these probabilities. Karpathy introduces `torch.multinomial`, a PyTorch function that returns indices sampled according to the provided probabilities. He uses `torch.Generator` with a fixed seed to make sure the reproducibility.   
He runs a test by sampling 20 items from a simple distribution. If the first item has a 60% probability, roughly 60% of the sampled integers turn out to be `0`.
![[Pasted image 20261009163320.png]]
```python
g = torch.Generator().manual_seed(2147483647)
# give me a probability distribution, and i will generate samples respect to the probability distribution
# replacement = True --> When a sample is drawn, it is "put back" into the pool.
# if you were asking for 10 sample, you could pick the same index multiple times.
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

itos[ix] #output: c
```
Karpathy puts everything together into a loop to generate entirely new names.
We first create the probability matrix `P`, we take the matrix of count `N` and divide every element in a row by the sum of that row. N is a tensor with shape 27 * 27. Every row represents a character and the probability of what will appear next to that character (Like the plot we created before).
```Python
# N + 1 guarantee the model smoothing; Because in the training data,
# Some character combination might never appear （e.g., "qz" never happens).
# If N has a 0 in that spot, the probability becomes 0. iF YOUR MODEL LATER ENCOUNTERS “qz” in a validation set, it will assign it infinite loss ( clamming it is impossible)
# "Add-one" Smoothing, it ensures no probability is ever exactly zero.
P = (N+1).float()
# without keepdim, the tensor become of shape [27], row sums become values such as 6.9246, etc..
# with keepdim = True, every row sums to 1.00
P /= P.sum(1, keepdim=True)
P.sum(1)
```
**Why `keepdim=1` are so important?** Without the `keepdim=TRUE`,  the shape of the tensor `P` will be `(27,)`, and pytorch right-aligns this shape and interprets it as `(1,27)`. Because `P` happens to be square, the operation succeeds --- but applies the division on **columns**, producing silently incorrect results. With `keepdim=True`, the shape of `P`becomes `(27,1)` and the division will be on the **row**. every row sums to 1.

The Loop:
* **The Process:**  
	  1. Start with the special "start token" (index 0).  
	  2. Fetch the row from the matrix corresponding to the current character.  
	  3. Normalize that row to get probabilities.  
	  4. Sample the *next* character index using `torch.multinomial`.  
	  5. Repeat until the "end token" (index 0\) is sampled.  
* **The Output:** He generates several names (e.g., "mor", "axx", "mo").  
* **Quality Check:** He admits the results are "terrible" because a Bigram model is too simple; it only looks at the single previous character without knowing the broader context of the word. However, he shows that compared to a completely random (uniform) model], the Bigram model is indeed generating more "name-like" sequences.
```PYTHON
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
  # List to hold the characters of the generated name
  out = []
  
  # set the current index to 0
  # In this dataset, the index 0 maps to the '.' character, which server as "Start Token" / "End Token"
  ix = 0

  while True:
    # Look up the row in the probability matrix corresponding to the current character (ix)
    # 'p' is now a list of probabilities for what the "next" character could be
    p = P[ix]

    # Draw a random sample from the distribution 'p'
    # The higher the probability in 'p', the more likely that index will be chosen.
    # item() converts the resulting tensor into a simple integer
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

    # Convert the integer index back into a character (using the lookup table 'itos')
    # and add it to our list of characters.
    out.append(itos[ix])

    # If the sampled index is 0 (the '.' character), it means the model has generated an "End Token".
    # We break the loop to stop generating this word.
    if ix == 0:

      break

  print(''.join(out))
```

**Evaluating Model Quality using Loss Function**. We need to summarize the model’s performance into a single number. We want a metric that tells us how good the model is at predicting the training data.   
The **Likelihood** is introduced, which is **product of the probabilities assigned to all correct next characters in the dataset**. A **good** model should have a **high** likelihood (product of probabilities close to 1). It works like this:
-  Imagine we have a Bigram model that has already been trained. It has **probability table** (The matrix 27 * 27) that tells us how likely any letter is to follow another. Now we want to measure how “good” this model is at predicting a specific name, like “emma”.
	- **Identify the “Correct next Characters”:** First, we break the word “emma” into the actual events (transitions) that occurred. These are the “correct” next characters because they are what actually exists in the data. For “emma”, the transitions are:
		1.  **.** (start) → **e**   
		2.  **e** → **m**  
		3. **m** → **m**  
		4. **m** → **a**  
		5. **a** → **.** (end)
	- **Find the “Assigned probabilities”**: Next, we look at our model’s probability matrix to see how much confidence it assigned to these specific transitions. 
		- *Model, what is the probability of e starting a word?* → Let's say **0.05** (5%)  
		* *Model, what is the probability of m following e?*  →  Let's say **0.04** (4%)  
		* *Model, what is the probability of m following m?*  → Let's say **0.02** (2%)  
		* *Model, what is the probability of a following m?* → Let's say **0.40** (40%)  
		* *Model, what is the probability of the name ending after a?*  → Let's say **0.30** (30%)
	* **Calculate the “Product”**: To find the likelihood of the entire name "emma" existing according to our model, we multiply these probabilities together. In probability theory, the probability of Event A **AND** Event B happening is ***P(A)\* P(B)***. Likelihood(“emma”) \= 0.05 \* 0.04 \* 0.02 \*0.40 \*0.30 \= 0.0000048.
* This calculation tells us **how "surprised" the model is by the data.**  
	* **If the product is high (close to 1):** The model assigned high probabilities to the correct letters. It "knew" what was coming. It is a **good model**.  
	* **If the product is low (close to 0):** The model assigned low probabilities to the correct letters (e.g., it thought `z` should follow `e`, but `m` showed up instead). The model was "surprised" by the reality. It is a **bad model**.
The problem of the likelihood is **multiplying many probabilities** (numbers between 0 and 1\) results in a **tiny number**. To fix this, he takes the **Log Likelihood** [[Log, Loglikelihood]].
- log(1) is 0  
- log(0.5) is \-0.3  
- log(0) is negative infinity.  
- The sum of log probabilities gives us a manageable negative number.
**Negative Log Likelihood**, since we typically want to **minimize** a loss function (where low is good) rather than maximize likelihood, he **inverts the sign** to create the Negative Log Likelihood. He further refines this by taking the average (dividing by the count of examples) so the loss is independent of the dataset size. He calculates the loss for the current Bigram Model on the entire training set, resulting in a loss value of approximately 2.45.

### **How log-likelihood is calculated across the full dataset and why lowering the Negative Log-Likelihood (NLL) directly translates to a better model?**
To get the Log-Likelihood for the entire dataset, we simply sum the log-probabilities of every single correct character transition that occurs in every name in the list. The process are:
1. **Iterate:** We loop through every name in the dataset (e.g., "emma", "olivia", "ava"...).  
2. **Extract Bigrams:** For each name, we look at every character pair `(ch1, ch2)`.  
3. **Lookup Probability:** We check the model's assigned probability for that transition: P(ch2|ch1)  
4. **Take the Log**: we calculate the natural logarithm: In(P)  
5. Sum: We add this value to a running total.
Mathematical Formula: if our dataset `D` has $N$ total bigrams, the Log-likelihood ($LL$) is: $$LL = \sum_{(ch1, ch2) \in D} \ln(P(ch2 \mid ch1))
$$
Because probabilities are always \<= 1, their logs are always negative (or zero). So, this sum will be a large **negative number.** 

- **High probability** (good): In(1.0) \= 0 → NLL \= 0 (lowest possible loss)  
- **Low Probability** (Bad): In(0.01) \= \-4.6 → NLL \= 4.6 (High Loss)

```python
# initialize a counter for the total number of bigrams (pairs) processed
n = 0

# iterate over every word in the dataset
for w in words:
#for w in ["andrejq"]: # optional debugging line to test a specific name

  # Add start / end tokens to form the sequence
  chs = ['.'] + list(w) + ['.']
  # Iterate through every bigram (pair of characters) in the word
  for ch1, ch2 in zip(chs, chs[1:]):

    # convert characters to integer indices to access the matrix
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]

    # Look up the probability assigned by the model to this specific transition
    # "Given we saw ch1, what was the probability of seeing ch2"
    prob = P[ix1, ix2]

  
    # Take the natural logarithm of the probability
    # if prob = 1.0, logprob = 0. if prob is small (e.g., 0.01), logprob is a large negative number
    logprob = torch.log(prob)
    # Add this log-probability to the running total
    log_likelihood += logprob
    # Increment the count of total bigrams processed
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

  

print(f'number of bigram processed {n}')

print(f'{log_likelihood=}')

nll = -log_likelihood

print(f'{nll=}')

print(f'{nll/n}')
```

## Part 2 (Neural network framework)
Instead of explicitly counting character pairs (the statistical approach), we cast the problem into a **Neural Network Framework**. We can build a network that **takes a single character** as input and outputs a **probability distribution** for the next character. While the counting method is simpler for a Bigram model, the neural network approach is significantly more **flexible** and **scalable**. It will allow us to eventually **expand the context** (looking at more than one previous character) without the “**state space explosion**” that occurs with simple tables.
- **State space** refers to the total number of possible scenarios (states) your model has to remember. **In the "Counting Table" Approach:** We build a table where every possible input has its own dedicated row.

### Building the Training set (`xs`, `ys`):
To train the network, We first constructs the training dataset using the word “emma” as a simple example.
- **Inputs (`xs`)**: The integer index of the current character.  
- **Target (`ys`)**: The integer index of the character that follows it.  
- **Example for “emma”:**  
	* Input: . (Start) →  Target: e  
	* Input: e  → Target: m  
	* Input: m → Target: m
```python
# create the training set of bigrams (x,y)

xs, ys = [], []

for w in words[:1]: #emma

  chs = ['.'] + list(w) + ['.']

  for ch1, ch2 in zip(chs, chs[1:]):
    # Convert the 1st character (input) to its integer index
    ix1 = stoi[ch1]

    # Convert the second character (Target) to its integer index
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs) # tensor([ 0, 5, 13, 13, 1])
ys = torch.tensor(ys) # tensor([ 5, 13, 13, 1, 0]) 
```
- when the input to the neural network is integer 0 → the desired label is integer 5 (which corresponds to e)  
- when the input to the neural network is 5 → we want its weights to be arranged so 13 gets a very high probability.  
- when 13 is put in, we want 13 to have a high probability and also 1 to have a good probability.   
- when 1 is put in, we want 0 to have a high probability etc..
We got one problem: that integers (like 13 for 'm') cannot be fed directly into a neural network because multiplying a weight by an index value (e.g., ***weight * 13*** ) implies a mathematical magnitude that doesn't exist for categorical data (letters). To understand this, we need to look at the basic math operation inside a neural network neuron. A neuron calculates its output using this formula: $$output = weight * input$$If we feed a **raw integers** (like 1 for 'a' and 26 for 'z') into this formula, we accidentally force the network to assume 2 things that are not true: 
- **Magnitude** (Size): For example, a 26 is twenty-six times bigger than the number 1. if we feed the network 26,  The network will react **26 times more intensely** to the letter '**z**' than to the letter '**a**', simply because '**z**' happens to be at the end of the alphabet. This is illogical. In the context of names, '**z**' is not "more" of a letter than '**a**'; it's just a *different* letter.  If the network tries to learn a weight to handle '**a**' correctly, that same weight will blow up the result for '**z**'. It becomes impossible to tune the weight to work for both characters because their input values are on vastly different scales. 
- **Order** (Distance): Integers imply a specific distance between concepts. 1 is close to 2 and it's far away from 26. If you use integers, you are telling the network that **'a' (1)** is very **similar** to **'b' (2)**, but **'a' (1)** is completely **different** from **'z' (26)**.
To fix this, we use **One-Hot Encoding**,  Instead of `a=1` and `b=2`, we use vectors where every letter is equally distinct and has the same "size" (magnitude).
Now, when we do the math ($Weight * Input$), every letter acts as an independent switch:
- If the input is **'a'**, it multiplies the first weight by 1 and ignores the rest. Because the one-hot encoded vector looks like this:   **a:** `[1, 0, 0, ...]`.
* If the input is **'z'**, it multiplies the last weight by 1 and ignores the rest.

This allows the network to learn a specific, independent weight for 'a' and a completely different, independent weight for 'z', without one interfering with the other.  
**PyTorch Implementation:**  Use `F.one_hot(xs, num_classes=27)` to automatically transform the integer inputs into these vectors. Since neural networks require floating-point numbers for matrix multiplication, he casts these integer vectors to float using `.float()`.
```python
import torch.nn.functional as F

# xs is training data for "emma", so it looks like this: tensor([ 0, 5, 13, 13, 1])
xenc = F.one_hot(xs, num_classes=27).float()

xenc
# Output: tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```

### Constructing a Linear Layer
#### Constructing a single neuron
**Goal:** Create a single neuron that processes the input vectors.
- **Weights (`w`):** A neuron is defined by its weights. Karpathy initializes these weights using `torch.randn`, which generates random numbers from a normal distribution (centered at 0, with some variance).
- **Shape:** For a single neuron accepting 27 inputs (the one-hot encoded characters), the weight matrix size is set to `(27, 1)`.
- **Operation:** The neuron's activation is calculated using matrix multiplication (`@` in Python). By multiplying the input matrix `x_enc` (shape `5x27`) by the weights `w` (shape `27x1`), the result is a `5x1` matrix. This represents the activation of that single neuron for all 5 input examples (the characters in "emma").
	- You can think of: A neuron as a “**scanner**” or “**filter**” that assigns a specific score to every letter in the alphabet.  THe “Activation” is simply the result of applying that “scanner” (the weights) to an input, mathematically, it is a dot product). Each of the 27 neurons represents
```python
# weight intialization for matrix W

W = torch.randn((27, 1))

xenc @ W
```

#### Scaling to a full Layer of Neurons
Instead of just one neuron, we want 27 neurons (one for each possible next character). The weight matrix w is resized to **(27,27)**.
- Rows (27): Correspond to the 27 input dimensions (the incoming character)  
- Columns (27): Correspond to the 27 distinct neurons.

**Difference between have 1 Neuron (`27*1`) and 27 neurons (`27*27`)**
The difference is the difference between asking “**What is the score for the letter ‘a’ being next**” vs “**What are the scores for every single letter in the alphabet being next?**”.
**The 1 Neuron Case (`27x1` Weights)**: When you have a weight matrix of `27*1`, you have one single filter.
* **The Math:** Input (`5x27`) `@` Weights (`27x1`) \= Output (`5x1`).  
* **The Result:** You get a single column of numbers.  
* **The Interpretation:** This single neuron is hard-coded to look for **one specific thing**. For example, let's say this specific neuron is responsible for detecting if the letter "z" comes next.  
- If you input "a", it gives you a score (e.g., 0.1). This score represents: *"How likely is it that 'z' comes after 'a'?"*  
* **The Problem:** This is **useless** for generating text because you don't just want to know if "z" is next. You need to know if "a" is next, or "b", or "c"... you need to compare **all** possibilities.
**The 27 Neurons Case** (`27x27` Weights):  When you expand the weights to `27x27`, you are creating **27 distinct filters** (27 neurons) running in parallel.
* **The Math:** Input (`5x27`) `@` Weights (`27x27`) \= Output (`5x27`).  
* **The Result:** You get 27 columns of numbers.  
* **The Interpretation:** Each column represents a different neuron with a different job.  
- **Column 0:** "I calculate the score for **'a'** coming next."  
- **Column 1:** "I calculate the score for **'b'** coming next."  
- ...  
- **Column 26:** "I calculate the score for **'.'** (end) coming next."

Matrix multiplication remains same `x_enc @ w`:
- **Input (`5x27`)** `@` **Weights (`27x27`)** `=` **Output (`5x27`)**.
The resulting `5*27` matrix contains the activations for every neuron on every input example. For instance, the value at index `[3,13]` （L） represents the firing rate of the 13 neurons (L) looking at the 3rd input example.
Karpathy emphasizes that matrix multiplication performs all the necessary dot products efficiently in parallel. He demonstrates this by manually calculating the dot product for a specific input-neuron pair (multiplying the input row by the weight column and summing) to show it matches the result from the matrix multiplication operation.
The resulting structure is a **linear layer**. It’s “Linear” because it performs the operation `W * x` without adding a bias term (`b`) or a non-linear activation function (like `tanh` or `sigmoid`).  This is the simplest possible neural network structure and serves as the foundation for the more complex models built later in the series.
```python
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
W = torch.randn((27, 27))
logits = xenc @ W # predict log-counts
```
Karpathy emphasizes that matrix multiplication performs all the necessary dot products efficiently in parallel. He demonstrates this by manually calculating the dot product for a specific input-neuron pair (multiplying the input row by the weight column and summing) to show it matches the result from the matrix operation.
```python
# matrix multiplication
matrix_operation = xenc @ W

# print the 3rd row 13 column
print( f'The element of 3rd row and 13 column: {matrix_operation[3,13]}') #0.03788243606686592

# dot product of 3rd row against 13th column
# you will get the same number from the matrix multiplication
# this prove the matrix multiplication is doing dot product
test = xenc[3]* W[:,13]

test.sum() #tensor(0.0379)
```

### Interpreting Neural Network’s Outputs
The network takes an input character (e.g., ‘a’), multiplies it by weights (W), and produces 27 numbers. The **problem** is these numbers can be anything – Positive or negative (e.g., \-2.5, 0.4, 8.1). We need **Probabilities**. Probabilities must be **positive** and probabilities **must sum to 1.0**. 
- To solve the “Positive number” problem, he introduces exponentiation ($e^x$).  
* **Logits** \= Raw output of the neural layer “logits” (log-counts).  
* **The math**: He applies the **exponential function** to these logits.  
	- If a logit is negative (e.g., -3), $e^{-3}$ becomes a small positive number (closer to 0).  
	- if a logit is a positive (e.g., +5), $e^{+5}$ becomes a large positive number.  
* **Result**: We now have a list of strictly positive numbers. These are interpreted as “Fake counts” – equivalent to the raw frequency counts we manually collected in the first method.
**Normalization (The probabilities)**: Now that we have “counts”, we need to turn them into probabilities.
- **The operation**: Just like with a manual table, **he sums up the counts for the row** and **divides each count by that sum**. Result is a valid probability distribution where all numbers are between 0 and 1, and the entire row sums to exactly 1.0.
These two steps (Exponentiation → Normalization) as the **Softmax** function. Softmax is a standard activation function that takes a vector of arbitrary real numbers and “squashes” them into a probability distribution. Formula: $$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$This makes the neural network’s output **compatible with our loss function** (Negative Log Likelihood) and allows us to train the model using gradient descent. The network effectively learns to predict “log counts” that maximize the likelihood of the correct next character.
```python
# xnec: One-hot encoded input
# w: weight matrix of shape (27,1)
# The result "logits" acts as the "Scores" for the next character
logits = xenc @ W # log-counts

# We take e^x for every number in logits
counts = logits.exp() # equivalent  (table of counts)

print(f'counts = {counts}')
print(f'counts shape = {counts.shape}')
print(f'counts.sum(1, keepdims = True) shape = {counts.sum(1, keepdims=True).shape}')
probs = counts / counts.sum(1, keepdims=True)
```
output: ![[Pasted image 20261016021101.png]]
The neural network is essentially attempting to “predict” the counting table we built manually earlier, but with a mathematical twist to make it learnable.  The comparison: Manual Table vs Neural Net:

|                  |                                                  |                                                                                            |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Step             | Method A: Manual Counting                        | Method B: Neural Network                                                                   |
| 1. Source        | You literally count the pairs in the text.       | The network calculates x @ W to get Logits.                                                |
| 2. The "Count"   | You get an integer (e.g., 50). Must be positive. | You apply exp(logits).<br><br>  <br><br>e^logit results in a positive number (e.g., 50.0). |
| 3. Normalization | Divide by the row sum.                           | Divide by the row sum.                                                                     |
| 4. Result        | Probabilities (e.g., 20%)                        | Probabilities (e.g., 20%)                                                                  |
**Why doesn’t the neural network just predict the probabilities directly?**  There are two main reasons why we predict “log-counts” logits:
**A. The “Negative Number” Problem:**  
Neural networks use matrix multiplication (`x @ W`). The weights in `W` can be **negative numbers** (and they start that way randomly). 
- If a neuron outputs \-5.0, you cannot interpret that as a count. You cannot see a letter “-5 times.”  
- You also cannot interpret it as a probability (probabilities must be >= 0).
By treating the output as a Log-count (Logarithm of the count), we solve this. 
- The network outputs \-5.0 (The Logit).  
- We exponentiate it: $e^{-5.0}$ \= 0.0067.  
- Now we have a valid, small positive “count”.

**B. Continuous vs. Discrete (The Learning aspect)**
* **The Manual Table** is "Discrete." You simply memorize the history. If 'b' followed 'a' 50 times, the count is exactly 50. You cannot change this; it is a fact of history.  
* **The Neural Net** is "Continuous." It doesn't memorize history; it tries to find a configuration of weights that **mimics** the history.  
- It outputs a score.  
- We compare the score to the real data.  
- If the score is wrong, we nudge the weights slightly using gradient descent

By predicting “Virtual counts” (Logits), the network has a **smooth dial** it can turn up and down to adjust the probabilities until they match the patterns in the data.
#### **In which dimension should softmax apply?**
In pytorch library the softmax function is class torch.nn.Softmax(dim=None) accepts **dim** argument. Depending on the **dim** argument, the softmax function will be applied to that specific dimension. So in which dimension should we apply softmax usually?

Usually, you normalize along the dimension that represents the **feature** or **classes**. A helpful rule is the “Sum to 1”Test: Ask yourself, *"Which set of numbers represents a complete set of mutually exclusive probabilities that should add up to 100%?"*
*In* almost all classification and language modeling tasks, the tensor is organized so that the last dimension holds the raw scores for every possible category.
- **Shape:** `(Batch_Size, Vocab_Size)` or `(Batch_Size, Sequence_Length, Vocab_Size)`  
- **Goal:** "Is this character 'a', 'b', or 'c'?"  
- **Action:** You want the scores for 'a', 'b', 'c' to sum to 1\. Since those scores live in the last dimension (`Vocab_Size`), you use `dim=-1`.

**Batch**, Represents different independent examples, you do not normalize here.  
**Sequence\_length** represents the sequence order, if you normalize here, you force the probability of the 1st word and the 2nd word to sum to 1\. Does it make sense? No. A sentence can have high probability words at step 1 and step 2\.  
**Vocab(C),** the alphabet(a,b,c…). If you normalize here, you force the model to choose: “if it’s likely an ‘a’, it’s less likely a ‘b’”. It does make sense. This is the prediction.

**Summary of the Neural Network Pipeline**: Karpathy consolidates all the individual components discussed so far into a single, clear "Forward Pass" summary.
- **Input:** He takes the integer indices of the characters (e.g., for "emma").  
- **Encoding:** He converts them into **One-Hot Vectors** (floating point numbers).  
- **Linear Layer:** These vectors are multiplied by the Weight Matrix `W` to produce **Logits**.  
- **Softmax:** The logits are exponentiated (to become positive "counts") and normalized (to sum to 1) to become **Probabilities**.  
- **Loss:** He calculates the **Negative Log Likelihood** by looking at the probability the network assigned to the *actual* next character.

The “Guess and Check” vs Gradient Descent: He shows that if you randomly change the weights W (re-sampling them), the loss changes (e.g., from 3.76 down to 3.37). You could just keep **guessing random weights** until you find good ones, but that is inefficient. We now have a **Loss Function that is fully differentiable**, meaning we can use **calculus (Gradient Descent)** to mathematically find the direction to move the weights to **minimize the loss.** Karpathy explicitly connects this process to his previous video series, **Micrograd**. The structure is identical:
- Define inputs and targets.  
- Perform a Forward Pass  
- Calculate loss  
- Zero the gradients  
- Perform a backward pass (`Loss.backward()`)  
- Update parameters (`data +=  - learning_rate * grad)`
In Micrograd, the network was a Multi-Layer Perceptron (MLP) with Mean Squared Error loss. Here, it is a single Linear Layer with Negative Log Likelihood loss. However, the **mechanism of optimization** is exactly the same.

### Training Process
**Forward Pass:**
- **The Goal:** He needs to calculate the loss for the entire dataset in a vectorized way (without loops).  
- **Selecting Probabilities:** The network outputs `probs` of shape `[5, 27]`. However, we only care about the probability assigned to the *correct* next character.  
- **Vectorized Indexing:** He demonstrates how to "pluck out" these specific probabilities using PyTorch indexing: `probs[torch.arange(5), ys]`.  
	* `torch.arange(5)` creates indices `[0, 1, 2, 3, 4]` (the row numbers).  
	* `ys` contains the correct column indices (targets).  
	* This combination selects the exact probability the model assigned to the correct answer for each example.  
- **Final Loss:** He takes the log of these probabilities, computes the mean, and negates it. The result is **3.76**, matching the manual calculation from earlier.
```python
# forward pass

xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding

logits = xenc @ W # predict log-counts

counts = logits.exp() # counts, equivalent to N

probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

# Vectorize way to calculate negative Log likelihood for the entire batch of data at once
# It replaces the for loop we wrote earlier.
# Instead of calculating the loss one character at a time,
# it does it for all 5 examples simultaneously.
# This is advanced indexing.
# It allows you to pick specific items out of matrix using coordinate pairs
# torch.arange(5) --> [0,1,2,3,4] = Row indices
# column indices: ys contians the correct targets [5,13,13,1,0]
# Pytorch will pairs them up: (0,5), (1,13), (2,13).., etc
# .log() converts the probabilities into Log-likelihood
# .mean(): Instead of just summing them (which creates a huge number if you have a million examples), we take the average.
# This allows the loss to stay consistent regardless of batch size.

loss = -probs[torch.arange(5), ys].log().mean()
```

**Backward pass**:
* **Zeroing Gradients:** He sets `W.grad = None`. This is efficiently interpreted by PyTorch as "zero gradients."  
* **`loss.backward()`:** This single command traverses the entire computational graph in reverse and fills `W.grad` with the gradients.  
- `backward()` computes gradients for every tensor that was involved calculating the loss, but only if that tensor has requires `_grad = True`. And in this specific Neural network (the bigram model), W (the weights) is indeed the only leaf node in the graph marked with requires`_Grad = True`.  
* **Interpretation:** He inspects `W.grad` (shape `27x27`). A positive value in a specific cell means that *increasing* that weight would *increase* the loss. A negative value means increasing the weight would *decrease* the loss.
```python
# backward pass
W.grad = None # set to zero the gradient
loss.backward()
```
**Update step**: He performs the single step of optimization (Gradient Descent).
- **Formula:** `W.data += -0.1 * W.grad`.  
* He uses a learning rate of `-0.1`.  
* He modifies the data in-place.  
- W.shape is torch.Size(\[27, 27\]) and W.grad shape is also torch.Size(\[27,27\]). Thats why we can update the W.data with W.grad.   
- **Result:** After one update, he re-runs the forward pass, and the loss drops from **3.76** to **3.74**. After a second update, it drops to **3.72**. This confirms the network is learning.
```python
W.data += -0.1 * W.grad
```
**Putting it all together**: He notes that the code requires zero changes to switch from 5 examples to 228,000 examples (the full dataset). He runs the loop for ~100 iterations. The loss rapidly decreases to around **2.47**. This is significant because it matches the loss achieved by the "counting method" earlier in the video. We proves that the gradient-based approach arrives at the exact same result as the statistical counting approach, but via a different, more flexible path.
```python
# gradient descent

for k in range(100):

  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding

  logits = xenc @ W # predict log-counts

  counts = logits.exp() # counts, equivalent to N

  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character

  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()

  print(loss.item())


  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -50 * W.grad
```
