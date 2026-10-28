**Recap for the limitations of the Bigram Model**: Bigram is a simple model we built in the last chapter; This simple model predicts the next character based only on the single preceding character using counts and probability normalization.  
While approachable, the bigram model produces poor predictions because it lacks context. If you try to increase the context (e.g., looking at 2 or 3 previous characters), the "lookup table" size explodes exponentially. For example, 3 characters of context would result in over 20,000 possibilities ($27 \times 27 \times 27 = 19,683$, the model looks at the **3 previous characters to predict the 4th one**), making it impossible to get enough data to populate the table effectively.  

### **MLP language model paper walkthrough**
To solve the context problem, we want to transition to building a multilayer perceptron (MLP) model, following the approach of the influential paper by Bengio et al. (2003). 
- **Embeddings:** The core idea is to associate each word (or character, in this video's case) with a feature vector. In the paper, 17,000 words are mapped to a 30-dimensional space.  
	* 30-dimensional space means a single vector with 30 elements. Looks like this:![[Pasted image 20261019224546.png]]
	* The matrix in the Original paper was structured as 17,000 (Rows) \* 30 (Columns). Think of this matrix exactly like a spreadsheet or a database table where you store information about employees.  
		- **Rows \= Items (Words):** You need one entry for every single word in your dictionary. If you have 17,000 words, you need 17,000 rows.  
		- **Columns \= Attributes (Dimensions):** You need to store the specific "stats" or features for each word. If you decided on 30 features, you need 30 columns.
	- The number 30 is a Hyperparameter. You choose it.  
		- **If too small (e.g., 2 dimensions):** The space is too cramped. The model can't capture complex relationships (like gender, tense, plurality, and synonymy) all at once. It would be like trying to describe a person using only "Height" and "Weight"—you miss too much detail.  
		- **If too large (e.g., 17,000 dimensions):** This defeats the purpose. The model would just memorize the words (One-Hot Encoding) rather than learning their meaning. It would also require massive computing power and huge amounts of data to train effectively without "overfitting."  
		- **Constraint of information:** By forcing 17,000 words to fit into a small 30-dimensional space, you force the model to be efficient. It *must* group similar words (like "blue," "red," "green") together to save space. This "compression" is what allows the model to generalize to new sentences.
	- **Training:** These feature vectors start as random but are tuned via backpropagation. Words with similar meanings or synonyms will naturally end up close to each other in this geometric space.
Why does embedding works?  Even if the model has never seen a exact sentence in the training data, it can still make a good prediction if it has seen a similar sentence like *"The cat was running in a..."*.  Because the network learns that "dog" and "cat" are similar (and thus have similar embedding vectors), **knowledge gained from one context can be transferred to unseen scenarios**.

#### **Look Up table**
The neural network way of creating a “look up” table is totally different from creating a look up table with characters. Like we did in the Bigram model, we create a linear layer with 27 neurons and take in input a vector of 27 elements and output the probability for the next character. In this part, we are going to create a Look Up table that is to serve as a **trainable translator** between the human world of characters and the mathematical world of the neural network.  
Three main purposes of the Look up table:
1. **To “featurize” the characters**: The neural network cannot understand that ‘a’ is similar to ‘e’ just by looking at their integer indexes (e.g., index 1 and index 5).  
	- **Purpose**: The table assigns a set of floating-point numbers (Features) to each character.  
	- **Result**: Instead of just being “Character \#5”, it becomes a vector like \[0.4, \-1.2\]. These numbers give the model “knobs” it can turn to learn specific properties about that character。  
2. **Dimensionality reduction**: “Compress the vocabulary”. This table is crucial for efficiency.  
	- **Without the table:** You would have to feed massive, mostly-empty "One-Hot" vectors (size 27\) into the network.  
	- **With the table:** You pass dense, compact vectors (size 2\) into the network. This drastically reduces the number of parameters the hidden layer needs to handle, making the model faster and easier to train.  
3. **To learn geometric relationships**: The most important purpose is that this table is **not fixed**—it is trained.  
	- It starts as random noise.  
	- As the model learns, it moves these vectors around in that 2-dimensional space.  
	- **The Goal:** The table eventually organizes itself so that characters that function similarly in language (like vowels `a, e, i, o, u`) end up grouped close together in the vector space. This allows the model to generalize (if it learns a rule for 'a', that rule automatically applies to 'e' because they are neighbors in the table).

Neural network architecture (that will be implemented):
- **Input layer**: The model takes previous words (indexes) as input, for 3 previous words, these indexes are used to look up their corresponding vectors from a shared look up table matrix ***C***.  
	* in the paper, the indexes ranging from 0 to 17,000, because it has 17,000 vocabulary in the training set.  
- **Hidden Layer:** The input vectors are fed into a hidden layer. The size of this layer is a **hyperparameter** (a design choice, e.g., 100 neurons). This layer is fully connected to the inputs and uses a `tanh` non-linearity  
- **Output Layer:** The final layer has a neuron for every possible next word (17,000 in the paper). It is fully connected to the hidden layer and outputs **logits.**  
- **Softmax:** These logits are exponentiated and normalized (Softmax) to produce a probability distribution for the next word, which is maximized using log-likelihood during training.
### **Build the dataset:**
```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

# X = Input to the neural network
# Y = Label
X_test, Y_test = [], []

for w in words[:5]:
  print(w)
  # Initialize the context window with "..." (represented by index 0)
  # This serves as the padding before the actual word starts
  context = [0] * block_size

  print(context) # will be [0,0,0]

  # iTERATE THROUGH EACH CHARACTER IN THE WORD, APPENDING "." at the end
  # The "." acts as the special "End of word" token
  for ch in w + '.':
    ix = stoi[ch] # Convert the current character to its integer
    X_test.append(context) # Append the current sliding window character
    Y_test.append(ix) # Append the current character (the prediction)

    print(''.join(itos[i] for i in context), '--->', itos[ix])
    # Update the context window for the next iteration:
    # Remove the first character (Crop) and add the new character index to the end
    context = context[1:] + [ix] # crop and append

X_tensor_test = torch.tensor(X_test)
Y_tensor_test = torch.tensor(Y_test)
```
For one iteration, the output looks like this:
![[Pasted image 20261020005452.png]]
`X_tensor_test` (input data) will look like this: ![[Pasted image 20261020005542.png|149]]
`Y_tensor_test` (label) will look like this: ![[Pasted image 20261020005611.png]]

### **Implementing the embedding lookup table:**

The **goal** of creating the Look Up table (Matrix C) is to **compress the vocabulary of 27 characters (**one-hot encoding vector with most 0s**)** into a **lower dimensional space**.  We creates Matrix C using random numbers (torch.randn). The matrix has **27 rows** (one for each character) and **2 columns** (the embedding size chosen for this specific demo). This acts as a table where each row contains the feature vector for a specific character index.
**Two way to Retrieve embeddings**: there are two mathematically equivalent ways to get a vector from this table, but one is significantly better for performance:
- Method A: One-Hot Encoding (Conceptual):  
	* You create a "one-hot" vector (a vector of all zeros with a single `1` at the character's index).  
	* You perform **matrix multiplication** between the one-hot vector and matrix `C`.  
	* notes: this works but is inefficient because multiplying by so many zeros is computationally wasteful   
- Method B: Direct Indexing (Practical)  
	* You simply use the integer index (e.g., `5`) to pull out the 5th row directly: `C[5]`.  
	* This is the preferred method because it avoids unnecessary computation and is much faster.

#### **Advanced Indexing with Pytorch:**
- **List Indexing:** You can pass a list of integers (e.g., `C[[5, 6, 7]]`) to get multiple rows at once.  
- **Tensor Indexing (Integer Array Indexing):** You can pass an entire tensor of integers. Think of it as a massive, simultaneous “Find and Replace” operation that preserves the shape of your data. Example:  
	* **Lookup table (Matrix C)** is very small for this example. It maps 3 characters (indices 0,1,2) to 2 dimensions. Matrix C:   
		- **Index 0:** `[0.1, 0.1]`  
		- **Index 1:** `[0.2, 0.2]`  
		- **Index 2:** `[0.3, 0.3]`  
	* Now imagine our **Input Tensor(X)**. In the video, this is a batch of data. Let’s say we have 2 examples, and each example has a context of 3 characters. shape of X = (2,3). $X = \begin{bmatrix} 0 & 2 & 1 \\ 2 & 0 & 0 \end{bmatrix}$, ![[Pasted image 20261021031819.png]]
	* The operation: `C[x]`. When you run `C[X]`, PyTorch looks at the structure of `X`. It visits every single "cell" in `X`, reads the integer inside, goes to Matrix `C`, grabs the corresponding vector, and **plugs it into that cell**.
	* The output tensor expands. Every single number in the original `2*3` grid has popped open to become a vector of size 2.  **New Shape:** `(2, 3, 2).` The resulting shape will be \[**shape of index tensor**\] (Input tensor X, here is (2,3)) \+ \[**shape of the element being retrieved**\] (The size of one row in matrix C, the size of one row is 2).![[Pasted image 20261021032030.png]]
	* **Batch Implementation:** The speaker takes the entire input data tensor `X` (which has a shape of `32x3`, representing 32 examples of 3 context characters each) and effectively plugs it into `C` like this: `C[X]`. 
		* PyTorch automatically retrieves the vector for every single integer in `X`.  
		* **Final Shape:** The output tensor has the shape **32x3x2**.  
			- `32`: The number of batches/examples.  
			- `3`: The number of characters in the context.  
			- `2`: The embedding dimension for each character.

**Another example of Integer Array Indexing, `C[Xb]`**
- Look Up table C’s shape = `[27,10]` → 27 letter (1 special case “.”) with embedding size of 10.   
- Batch of words that need to Look up the look up table with shape `[32,3]`, `Xb`
Logically what happens with integer array indexing?  Imagine you have a batch of words `Xb`, `batch_size = 32`, and with a context window of 3, this is just a grid containing **96** integers total  (`32*3`). Imagine Pytorch temporarily flattens your `Xb` tensor into one long list of 96 numbers. `[32,3]` → flatten → `[96]`. Now pytorch does exactly what List Indexing will do (e.g., `C[[5, 6, 7]]`). It takes that list of 96 integers and looks up the embedding vector (size 10) for each one.
- Input: `[96]` (integers)  
- Output: `[96,10]` (vectors)
And then Pytorch looks at the original shape of `Xb ([32,3])` and says, “Okay, the user wanted these 96 vectors organized into a grid of 32 by 3.” . It takes the `[96,10]` result and folds it back up. `[96,10]` → reshape → `[32,3,10]`. 
- 96 never changes, because it’s the count. Does the content change? Yes. The integers (indices) are replaced by the actual data (Vectors)
We replace by Row because it’s following the **Golden rule of Indexing** in python, numpy, and pytorch. Because in a 2D tensor,  Dimension 0 is the rows, and Dimension 1 is the columns. When you write `C[ix]`, you are providing indices for the 1st dimension only. Therefore, pytorch assumes you want to slice along the rows.
- **Get Rows (Default):** `C[ix]`  → "Pick these indices from Dim 0."  
- **Get Columns:** `C[:, ix]`  → "Take **all** items from Dim 0 (:), and *then* pick these indices from Dim 1."

### **Implementing the hidden layer and the associated Pytorch internals**
The **objective** is to **feed the embeddings into the hidden layer** of the neural network.
- **Initialization**: Initializes the hidden layer weights `W1` and biases `b1` randomly.  
	* `W1` size: 6 * 100 ( 6 inputs → 100 hidden neurons).  
	* `b1` size: 100 (biases for each neuron).  
- The **input** to this layer should be the embeddings. Since we have **3 context words** each with **2-dimensional embedding**, the total number of input features is **6** ($3*2$).

**The problem: Shape Mismatch:** Standard matrix multiplication (`input @ W1`) fails because of a dimension mismatch.
- **Current shape**: The embedding tensor `emb` is **3D** with the shape (32, 3, 2) (Batch Size * Context Length * Embedding Dim).  
- **Required Shape:** To multiply with `W1`, we need a **2D** matrix where all the features for a single example are flattened into one row. We need the shape `(32, 6).`   
	* **Solution 1**: Concatenation (Inefficient): a naive way to flatten the tensor:  
		- **Manual Slicing:** He manually grabs slices of the tensor (e.g., `emb[:, 0, :]` this gives you the entire first row) and uses `torch.cat` to stitch them together side-by-side.  
		- **`torch.unbind`:** To avoid hard-coding indices (which breaks if you change the context length), he uses `torch.unbind` to slice the tensor dynamically
			- **avoid this:** This method creates entirely new tensors in memory, making it computationally inefficient
	- **Solution 2**: View ( Efficient & Preferred): The standard way to reshape tensors in PyTorch is using the `.view()` function.
		- **The Syntax:** `emb.view(32, 6)` instantly reshapes the tensor into the desired format.  
		- **Flexibility:** You can use `-1` (e.g., `emb.view(-1, 6)`), which tells PyTorch to "infer" the first dimension (the batch size) automatically.
#### **Pytorch Internals → Storage vs View:**

* **Storage:** No matter the shape of a tensor (2D, 3D, etc.), the actual numbers are stored in computer memory as a **1D contiguous array (vector).**  
* **Views:** A "view" is just a logical representation. It consists of **metadata** (like `storage_offset`, `stride`, and `shape`) that tells PyTorch how to interpret that 1D array.  
* **Efficiency:** When you call `.view()`, PyTorch **does not** copy or move any data in memory. It simply changes the metadata (strides) to "look" at the same data differently. This makes the operation extremely fast.
With the input reshaped, the full hidden layer is implemented: `h = torch.tanh(emb.view(-1, 6) @ W1 + b1)`.  Adding the bias **b1** (shape **100**) to the matrix multiplication result (shape **32,100**) works via **Broadcasting**. Pytorch effectively **aligns the bias to the right,** (shape 100 becomes (1,100), then by broadcasting rule it will expand the bias to the shape (32,100) .) creates a “fake” dimension to match the batch size, and adds the bias vector to every single row of the batch.

### **implementing the output layer**
The final layer of the neural network which will produce the raw scores (logits) for the next character prediction.
* Weights (`W2`) & Biases (`b2`):  
* **Calculation:** The logits are calculated using the formula `h @ W2 + b2`.  
	* `h` is the output of the hidden layer after the activation function. `h = torch.tanh(emb.view(-1, 6) @ W1 + b1)`
* **Result:** This produces a tensor of shape `32x27` (32 examples, 27 character scores each).
To interpret the logits as probabilities, we manually implements the Softmax function:
- **Exponentiation:** `counts = logits.exp()` creates "fake counts" which must be positive.  
- **Normalization:** `prob = counts / counts.sum(1, keepdims=True)` ensures each row sums to 1\.  
- **Result:** A probability distribution `32x27` where each row represents the likelihood of the next character being `a`, `b`, `c`, etc.

**Calculating the Loss**: The model needs a single number to measure performance (loss). We retrieves the specific probability the model assigned to the true target for every example in the dataset. `prob[X.shape[0], y\]`.  The loss is calculated as the negative average log probability `-prob.log().mean()`. This manual implementation results in a loss value (e.g., ~17.0 initially) that needs to be minimized.

```python
# initialize parameters
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

# forward pass
emb = C[X] # (228146, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (228146, 100)
logits = h @ W2 + b2 # (228146, 27)

# softmax
# probability of each words appears after a specific context ("3 letters for example")
counts = logits.exp()
prob = counts / counts.sum(1, keepdim=True)
loss = -prob[torch.arange(X.shape[0]), Y].log().mean()

loss
```

Andrej introduces `torch.nn.functional.cross_entropy` as the standard, professional replacement for the manual steps above. Instead of manually calculating softmax → Log → NLL, you simply call `F.cross_entropy(logits, y)`. He demonstrates that this function produces the exact same loss value as the manual method but is preferred for efficiency and stability. 
Why we use `F.cross_entropy`?  There are two critical reasons to use the built-in function:
- **Efficiency**: It uses “fused kernels,” meaning it calculates the entire operation in one go without creating unnecessary intermediate tensors in memory (like counts or probs). It **simplifies the backward pass** (gradient calculation) mathematically, making training faster.  
- **Numerical stability**:  If logits are very large positive numbers (e.g., 100), `exp(100)` explodes to Infinity (`NaN`), causing the training to **crash**. `F.cross_entropy` handles this **internally by subtracting the maximum value from the logits before exponentiation** (because of the normalization we used here, we can actually **offset logits by any arbitrary constant value** that we want, we add any value we want, and we will always get the exactly result). This mathematical trick ensures the numbers never get too large (overflow) while preserving the exact same probability distribution.
Cross-Entropy and Negative Log Likelihood(NLL) are mathematically **identical** only when the **ground truth is a “Hard Label”** (meaning we are 100% sure of the correct answer).  They stop being the same (or rather, NLL stops being a sufficient description) when your ground truth is a **"Soft Label"** (a probability distribution rather than a single fact).
1. **The standard Case**: They are same. In most classification tasks, the target is a “one-hot” vector.  
	* Scenario: The image is a Cat.  
	* **Truth (p):** \[1.0, 0.0, 0.0\] (100% Cat, 0% Dog, 0% Bird).  
	* **Prediction (q):** \[0.8, 0.1, 0.1\].  
	* The Cross-Entropy formula is: $H(p,q) = -\sum_{i} p(i) \log q(i)$  
	* Since $p(i)$ is 0 for everything expect the cat, the sum collapses to a single term: $$\begin{aligned}
H(p,q) &= -(1.0 \times \log(0.8) + 0 \times \dots + 0 \times \dots) \\[1em]
H(p,q) &= -\log(0.8)
\end{aligned}$$
	- The result $-log(0.8)$ is exactly the negative Log likelihood of the correct class.
2. **The Exception "Soft Labels"**: There are advanced training techniques where the ground truth is **not** 100% certain. In these cases, you must use the full Cross-Entropy formula; simple NLL (looking at just one index) no longer captures the full picture.
	* Context A: **Knowledge Distillation**
	In this technique, a small "Student" model tries to learn from a massive "Teacher" model. The Teacher doesn't just say "It's a Cat"; it gives hints like "It's mostly a Cat, but it looks 5% like a Dog."
		- **Truth (Teacher's output):** `[0.90, 0.09, 0.01]` (Cat, Dog, Bird).
		- **Prediction (Student):** `[0.80, 0.15, 0.05]`.

	Here, the Cross-Entropy loss calculates the error across **all classes** because the student needs to learn the "0.09 Dog-ness" as well as the "0.90 Cat-ness." $$H(p,q)=-(0.90\log 0.80+0.09\log 0.15+0.01\log 0.05)$$If you just used standard NLL here (focusing only on "Cat"), you would ignore the valuable information that the teacher thinks "Dog" is more likely than "Bird."
	* Context B: Label Smoothing
		Sometimes, to prevent overfitting, we tell the model: "Don't be too confident." Instead of a target of `1.0`, we use `0.9` and spread the remaining `0.1` across other classes.
		- **Truth:** `[0.9, 0.05, 0.05]`
		- **Prediction:** `[0.8, 0.1, 0.1]`
		Again, Cross-Entropy sums the error from all three positions. Simple NLL would only look at the first position.
FOR more Cross Entropy, look here: [[Cross-Entropy Loss - Binary Cross-Entropy]]

### **Setting up the training Loop**
The training loop consists of 3 parts: Forward pass, backward pass, update. 
- **The Loop:** He creates a loop that runs for a set number of steps (e.g., 1000).  
- **Zero Gradient:** Inside the loop, `p.grad = None` is called first to reset gradients from the previous step.  
- **Backward & Update:** He calls `loss.backward()` to calculate gradients and then updates the parameters using `data += -lr * data.grad`.  
- **Result:** The loss decreases significantly (from \~17 to \~0.2), indicating the model is learning.
```python
# training on full dataset
for _ in range(10):
    # forward pass
    emb = C[X] # (228146, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (228146, 100)
    logits = h @ W2 + b2 # (228146, 27)

    # calcualte the loss
    loss = F.cross_entropy(logits, Y)
    print(loss.item())

    #backward pass
    for p in parameters:
        # zero gradient
        p.grad = None

    loss.backward()
    #update
    for p in parameters:
        p.data += -0.1 * p.grad
```
We did this training loop on a **small batch** for **debugging/sanity checks** to see if we made some mistake. The loss drops very low (close to zero) because the model is only training on 32 examples with 3,400 parameters. It is simply memorizing the data.  **Why not exactly zero**? The loss doesn’t hit perfect zero because of “collisions” in the data. For the same input (e.g., …(Starting point)), the training set might have multiple different correct next characters (e.g., e, o, i). The model cannot predict all of them with 100% certainty simultaneously, so some loss remains.

**Mini-BATCH optimization**: When switching to the full dataset (228,000 examples), the training becomes too slow because every step requires processing all examples.
* The solution: Implement **Mini-Batches**.  
* **Mechanism:**  
	- Generate random indices: `ix = torch.randint(0, X.shape[0], (32,))` to pick 32 random examples.  
	- Forward/Backward pass is done **only** on these 32 examples.  
* **Trade-off:** The gradient direction is now an approximation (noisy) rather than the exact true gradient. However, because you can take thousands of steps in the time it takes to do one full step, the model converges much faster overall.

### **Find a good learning rate**
To find a methodical way to choose a learning rate, Andrej first demonstrates the failure modes of bad learning rates:
* **Too Low**: The loss decreases, but extremely slowly. It would take forever to converge.  
* **Too High**: The loss “explodes” or becomes unstable (e.g., fluctuating wildly between high numbers).

The “Learning Rater Finder”: To find the optimal rate efficiently, the speaker implements a technique (popularized by the fast.ai library/Cyclical Learning Rates paper) to **sweep through a range of learning rates during a single run**.
- **Range**: He decides to check learning rates between **0.001 (10^-3) and 1.0 (10^0)**.  
- **Stepping in Log Space**: He notes that stepping linearly (0.001, 0.002, 0.003…) is not useful because it takes too much time in large numbers. Instead, he steps **exponentially**:  
	* He creates 1,000 exponents linearly spaced between **\-3** and **0**.  
	* The actual learning rates are calculated as 10^exponent.
```python
# Create 1000 exponents equally spaced between -3 and 0
# -3 corresponds to 10^-3 (0.001)
# 0 corresponds to 10^0 (1.0)
lre = torch.linspace(-3,0,1000)

# Calculate the actual Learning Rates using these exponents
lrs = 10**lre # cretes a tensor with same shape of lre
# while elements contains 10**lre
```
- `lre`: ![[Pasted image 20261027010825.png]]
- The Sweep: he runs the training loop for 1,000 iterations.  
	* At step 0, the learning rate is very low (0.001)  
	* At step 1000, the learning rate is very high (1.0).  
	* He records the loss and the learning rate used at every single step.
```python
# Lists to store the history so we can plot them later
lri = []   # To store the Learning Rate exponent used
lossi = [] # To store the Loss resulting from that Learning Rate

for i in range(1000):
    # 1. Mini-batch construction (Randomly pick 32 examples)
    ix = torch.randint(0, X.shape[0], (32,))

    # 2. Forward Pass
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    # 3. Backward Pass (Zero grad, then Calculate grad)
    for p in parameters:
        p.grad = None
    loss.backward()
  
    # 4. Update Parameters DYNAMICALLY
    # Use the specific learning rate for this step (i)
    lr = lrs[i] #lrs is a list of learinig rate ranging from 0.001 to 1

    for p in parameters:
        p.data += -lr * p.grad

    # 5. Record the stats
    lri.append(lre[i])       # Record the exponent (x-axis)
    lossi.append(loss.item()) # Record the loss (y-axis)
```
He plots the **Loss (y-axis)** against the **Learning rate (x-axis)** (specifically the exponent). The shape of this plot tells you exactly which rate to pick:![[Pasted image 20261027010839.png]]
- **Flat Region (Left)**: The Learning rate is too low; the loss isn’t improving.  
* **Explosion Region (Right):** The loss shoots up because the steps are too big and the model is diverging.  
* **The "Valley":** There is a specific region where the loss drops steepest (Around \-1.0 ) before it becomes unstable.  
* **The Selection Rule:** You should pick a value in the "valley," slightly **before** the lowest point where it starts to become unstable.
	* Here the valley is around 10^-1 (0.1). This confirms his initial guess was actually optimal.

### Split the dataset and Why?
As you make a neural network larger (more parameters), it becomes capable of simply **memorizing** the training data. This would result in a near-zero loss on the training data but terrible performance on new, unseen words. We don't want the model to memorize; we want it to **generalize**. We want it to learn the *patterns* of the language (like "vowels usually follow consonants") so it can create new, valid names it hasn't seen before.  
To ensure generalization, the standard practice is to split the data into **three distinct parts**, usually with an 80%, 10%, 10% ratio.
- **Training Split (80%)**: Used to **optimize the parameters** (Weights and Biases) of the network using gradient descent.  
- **Dev / Validation Split (10%):** Used to **optimize the hyperparameters**. Hyperparameters are the design choices you make *before* training (e.g., number of neurons, embedding size, learning rate). You train different versions of the model and keep the one that performs best on this validation set.  
- **Test Split (10%):** Used **only at the very end** to evaluate the final performance. You are not allowed to look at this during training or tuning. This protects against "overfitting to the validation set" and gives a true estimate of how the model performs on unseen data.
Andrej refactors the code to shuffle the dataset and create these three distinct buckets (`Xtr/Ytr`, `Xdev/Ydev`, `Xte/Yte`).
```python
# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?


def build_dataset(words):
  X, Y = [], []
  for w in words:
    #print(w)
    context = [0] * block_size

    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)

      #print(''.join(itos[i] for i in context), '--->', itos[ix])

      context = context[1:] + [ix] # crop and append

  

  X = torch.tensor(X)

  Y = torch.tensor(Y)

  print(X.shape, Y.shape)

  return X, Y

  

import random

random.seed(42)

random.shuffle(words)

n1 = int(0.8*len(words))

n2 = int(0.9*len(words))

  

Xtr, Ytr = build_dataset(words[:n1])

Xdev, Ydev = build_dataset(words[n1:n2])

Xte, Yte = build_dataset(words[n2:])
```
After training on the new split, he compares the **Training Loss** vs. **Validation Loss**. They are roughly equal (~2.23).  The model is **Underfitting**. Since the **training loss isn't significantly lower than the validation loss**, the model is not "smart" enough to even memorize the training data. This suggests the network is too small (too few parameters) for the task. To fix the underfitting, Andrej **increases** the model's **capacity** (size):
- **Hidden Layer:** Increased from 100 neurons to 200 neurons.  
- **Embedding** **Size**: Increased from 2 dimensions to 10 dimensions (Fixing the “bottleneck where 2D was too cramped to represent character relationship).  
- **Training** **Loop**: He runs the training for significantly longer (200,000 steps) and introduces **Learning Rate Decay** (lowering the learning rate by 10x in the second half of training) to allow the model to settle into a better result.
```python
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C = torch.randn((27, 2), generator=g)

W1 = torch.randn((6, 200), generator=g)

b1 = torch.randn(200, generator=g)

W2 = torch.randn((200, 27), generator=g)

b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters) # number of parameters in total, 
```
**Final Result:** The validation loss drops to **\~2.17**, which is a significant improvement over the initial Bigram model (~2.45).
One thing that while training, he plots the loss logs. The plotted line looks very thick and fuzzy. This is because of the **Mini-batch** training. Some batches of 32 names are "hard" (loss goes up), and some are "easy" (loss goes down). This creates noise. Despite the noise, the average of the line (the trend) clearly goes down, especially after the learning rate decay creates a visible “Step” down in the loss.
![[Pasted image 20261028015612.png]]
1. **Divide the dataset into training, validation and test.**  
2. **The Loop Experimentation (99% of the work)**: This is the phase where you spend almost all your time. to find the best hyperparameters for the model.  
	- **Step A (Train):** You pick a set of hyperparameters (e.g., Learning Rate \= 0.1, Hidden Neurons \= 100). You let the model study **only the Training Set**. It adjusts its weights to minimize the loss on this data.  
	- **Step B (Evaluate):** Once the model is trained, you pause. You give it the **Validation Set** (the practice exam).  
	- **Step C (Compare):** You look at the score (Loss).  
		* *Experiment 1 (LR=0.1):* Validation Loss \= 2.4  
		* *Experiment 2 (LR=0.01):* Validation Loss \= 2.2  
		* *Experiment 3 (LR=0.001):* Validation Loss \= 2.5  
	- **Step D (Select):** You see that Experiment 2 worked best. You discard the others.  
	- **Crucial Point:** During this phase, you are effectively "overfitting" the hyperparameters to the Validation Set. You are tweaking the settings until they get a high score on *specifically* the Validation data.  
3. **The danger zone, why we need the testset?** Because you kept tweaking the model until it got a good score on the validation set, The **validation score is now biased**. It is likely an optimistic estimate. The model might have just gotten lucky with the specific questions in the practice exam. If you published the Validation score, you would be lying about how good your model actually is in the real world.  
4. The Final Step: The **Test** (1% of the work): This happens **only once**, right before you ship the product or publish the research paper.   
	- You take your absolute best model (from the Experimentation phase).  
	- You run it against the **Test Set** (The Final Exam).  
	- **The Rule:** You look at the score. **You accept the score.**  
		* If the score is bad, **you cannot go back and fix it.**  
		* If you go back and tweak the model to improve the Test score, the Test Set effectively becomes a *new* Validation Set. You have cheated (this is called "Data Leakage").