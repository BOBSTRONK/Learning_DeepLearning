### **Adjusting the initial Loss**
Notices that the loss at iteration 0 is **27**, which is identified as “way too high”. He calculates what the loss should be. With 27 possible characters (A-Z plus a special character “.”), a properly initialized network should assign roughly equal probability to each. 
- Probability per character \= 1/27  
- Expected Loss \= Negative log likelihood of that probability ( \-log(1/27)), which equals approximately 3.29.
The difference between the actual loss 27 and expected loss 3.29 implies the network is “confidently wrong” at initialization –assigning very high probabilities to the wrong characters and very low probability to the correct ones.  Because we initialize weights with random numbers and these random numbers contains the extreme number. Like **36.4366 in the first row of the tensor, this will causes high loss.**
![[Pasted image 20261030014917.png]]
Andrej demonstrates that when the network starts with such high loss, the loss curve looks like a “hockey stick” – it **drops sharply in the first few thousand iterations before flattening out.** This indicates the network is **wasting the first part of training** just **squashing the weights down to zero** to get to a **uniform distribution**, rather than actually learning the data relationships.
![[Pasted image 20261030014948.png]]
To fix this, Andrej manually adjusts the initialization of the last layer (logits):
- **Biases:** The biases (`b2`) are responsible for shifting the output. He sets them to **zero** (or very close to it) so they don't add arbitrary large values to the logits.  
- **Weights:** The logits are calculated as `h * W2 + b2`. Since `h` (hidden states) and `W2` (weights) are currently randomized, they produce large numbers. He fixes this by **scaling down the weights** (`W2`) significantly.  
	* He multiplies the initial random weights by **0.01** to squash them towards zero  
	* **Why not set to exactly zero?** He notes that while setting weights to exactly zero would give the perfect 3.29 loss, it is safer to leave a tiny amount of entropy (randomness) for symmetry breaking
After re-initializing with scaled weights and zero bias, the initial loss drops to roughly **3.3**, matching the theoretical expectation. The "hockey stick" shape disappears from the loss plot. The network no longer wastes time "squashing" weights and starts learning the actual task immediately.  This seemingly small fix results in a better final loss (improving from \~2.16 to \~2.12) because the entire training budget is spent effectively.
![[Pasted image 20261030015100.png]]
### **saturated tanh activations**
Andrej scrutinizes (examines carefully) the hidden state activation (h) – the output of the tanh function. 
- **Visualizing Values:** When he inspects the values of `h`, he notes that many of them are exactly **\-1.0** or **1.0**.  
- **Histogram Analysis:** He plots a histogram of the `h` tensor. Instead of a nice gaussian-like spread, the distribution is **bimodal**, with massive spikes at -1 and 1.![[Pasted image 20261030015436.png]]
- **The Cause:** He looks at the "pre-activations" (the values entering the tanh function). Because the weights and biases are initialized with standard random values, the pre-activations have a very broad range (e.g., -5 to +15). `tanh` squashes these large numbers into the flat "tails" of the function (very close to -1 or 1).
#### **tanh neuron can “only ever decrease” (or at best stay the same)**
The gradient of tanh is calculated like this: $\frac{dy}{dx} = 1 - y^2$, since `tanh(y)` always outputs a value between -1 and 1, let's look at what happens to the derivative $1-y^2$:
- Maximum value: if y \= 0, the derivative is 1 \- 0 \= 1\.  
* **Typical Value:** If ***y*** is anything else (e.g., **0.5**), the derivative is less than 1 (e.g., ***1 - 0.5 = 0.5***).  
* **Minimum Value:** If y is saturated at 1 or -1, the derivative is 1 - 1 = 0.
During backpropagation, the gradient “flowing” through the network is just a chain of multiplications. When the gradient arrives at a `tanh` gate from a later layer, it is **multiplied** by the local derivative of that `tanh` gate. $Gradient_{out} = Gradient_{in} \times Local \ Derivative$.
Because the Local derivative is always Between 0 and 1:
- You are always multiplying the incoming signal by a number like ***0.9,*** ***0.5***, or ***0.01***.  
- Mathematically, multiplying a number by a factor <= 1 will always result in a number with an equal or smaller magnitude.
This property is why tanh (and sigmoid) fell out of favor for very deep networks compared to ReLU:
- **Tanh:** Passes **\<100%** of the gradient. (The signal slowly decays as it goes back).  
- **ReLU:** Passes exactly **100%** of the gradient (if the unit is active) or **0%** (if inactive). It does not "scale down" the gradient, allowing deep networks to learn without the signal vanishing.   
	* ReLU is like a IF statement (a switch) that wither lets the signal go through perfectly ($\times1$) or kills it completely ($\times0$). The IF condition look at the original input (x) from the forward pass (the input of current layer / the output of previous layer).   
	* ReLU is also more efficient than TanH or sigmoid because its a IF statement and do not require the calculation like tanh or sigmoid

#### **Why is saturation bad?**
The Local gradient of a tanh neuron is calculated as ($1-tanh^2$), where t is the output activation. if the output is 1 or -1, the gradient becomes $1 - 1^2 = 0$. This **kills the gradient**. by kills the gradient it means it effectively stops the flow of information. The gradient flowing back from the loss function gets ==🟢multiplied by zero==, meaning the **weights connected to that neuron receive no update**. 
- **“Flat Region”**: Intuitively, if a neuron is in the flat tail of the tanh curve, changing the input slightly makes almost no difference to the output. Therefore, the network doesn’t know how to adjust the weights to improve the loss.![[Pasted image 20261031022842.png]]
The danger of **dead neurons**, which can happen in various activation functions (Tanh, ReLU, Sigmoid):
- **Definition:** A neuron is "dead" if it is saturated (e.g., outputs \-1 or 1 for Tanh, or 0 for ReLU) for **every single example** in the training set.  
- **Consequence:** If a neuron **never activates in its linear region**, it never receives a non-zero gradient. It essentially suffers "permanent brain damage" and will never learn or contribute to the network.  
		- If the slope is 0, the update formula becomes: `New_Weight = Old_Weight - (0)`. The weight never changes.  
- **Visualization Check:** He creates a boolean visualization (**white = saturated, black = active**) to check for dead neurons. While many neurons are saturated (white) for *some* examples, he verifies that no single column is *entirely* white (dead).
```python
plt.figure(figsize=(20,10))
plt.imshow(h.abs()>0.99,cmap='gray',interpolation="nearest")
```
![[Pasted image 20261031023140.png]]

#### **The fix: Scaling initialization**
To solve this, the **pre-activations need to be smaller** so they land in the linear, active region of the `tanh` function (roughly between -1 and 1). If the pre-activation ($x*W+b$) is a number between -1 and 1. the curve is steep and the gradient is strong. If pre-activation is large (5.0 or - 10.0) it falls in the “Saturated” Zone. The output is stuck at 1.0 or -1.0 and **the gradients drops to zero**.
```python
  

g = torch.Generator().manual_seed(2147483647) # for reproducibility

C  = torch.randn((vocab_size, n_embd),            generator=g)

W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)  * 0.1

b1 = torch.randn(n_hidden,                        generator=g)  * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) 

b2 = torch.randn(vocab_size,                      generator=g)  

```
- **Adjusting Biases (`b1`):** He sets the bias initialization to very small numbers or zero (multiplying by 0.01) to get a little bit of entropy, just so that there’s like a little bit of variation and diversity in the original initialization.  
- **Scaling Weights (`W1`):** The primary fix is to **scale down** the random weights. He multiplies the initial random weights by a small factor (e.g., **0.2**).  
- **Resulting Distribution:** After this change, the histogram of `h` looks much better—it is no longer concentrated at -1 and 1 but spread out across the range. The "white" saturation in his visualization disappears, meaning gradients can flow through all neurons.
**Result:** The distribution `h` changed from:
- Before:![[Pasted image 20261031023409.png|left|252]]
- After:![[Pasted image 20261031023440.png|left|315]]
And the pre-activation is changed from:
- Before:  Between -10 to 10 ![[Pasted image 20261031023520.png|left|260]]
- After: between -1.0 and 1 ![[Pasted image 20261031023647.png|left|285]]
And no-saturated region for h, the figure we plot before for visualization (black/white), now are full black (active).
![[Pasted image 20261031023736.png]]

All these changes are when we apply `0.1` to the W1. actually this is not that good. Andrej preferred the **0.2 scaling factor** over **0.1 scaling factor** because of a balance between non-linearity and gradient flow.
Look at ![[Pasted image 20261031024232.png]], it's   narrow “bell curve” concentrated tightly around 0. Most values are between $-0.5$ and $0.5$. The `tanh` function is nearly a straight line near zero (tanh(x) ≈ x ![[Pasted image 20261031024846.png]]). If your weights are too small (0.1), your pre-activation stay very small. The neurons effectively become **linear function**..
- If a neural network is just a stack of linear functions, it mathematically collapses into a single linear layer. You lose the "deep learning" capability because the network can no longer model complex, non-linear patterns.  
- Also, the signal "shrinks" as it passes through layers, leading to vanishing gradients in deeper networks.
After changing scaling factor from 0.1 to 0.2 → ![[Pasted image 20261031024525.png|left|183]]. The Distribution is much flatter and wider. It spreads significantly towards **-1.0** and **1.0**, covering the whole range. By boosting the weights (0.2), the pre-activations get larger. They are pushed further out from zero, entering the "curved" parts of the `tanh` function. It’s better. Because the values reach the curved areas, the network is actually using the non-linear properties of `tanh`. This allows it to learn complex boundaries. Crucially, while the distribution spreads out, it does not pile up into spikes at the very edges (-1 and 1). This means the neurons are active (non-linear) but not "dead" (saturated).
After the change we also get visualization of no-saturated region also has changed:![[Pasted image 20261031024927.png]]. And we can see it has some white points and it has some point it’s white (saturated).

After the change, we got:
- **Loss Improvement:** Fixing the saturated tanh improves the validation loss further, dropping it from roughly **2.13** to **2.10**.  
- **Why it Matters:** While a shallow network (like this 1-layer MLP) can sometimes recover from bad initialization by slowly learning its way out of saturation, **deep networks** often cannot. In deep networks, **bad initialization can lead to vanishing gradients across many layers**, causing training to fail completely.
The change that has ben applied:
```python
# MLP revisited`
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647) # for reproducibility

C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)  * 0.2
b1 = torch.randn(n_hidden,                        generator=g)  * 0.01

W2 = torch.randn((n_hidden, vocab_size),          generator=g)  * 0.1 # to initialize and make weights smaller center to 0 to avoid high loss in early training`
b2 = torch.randn(vocab_size,                      generator=g)  * 0
```

### How to calculate the init scale number?
While the manual scaling factor of 0.2 worked for this simple shallow network, it is a “magic number” derived from **guessing** and **checking**. This has the **scalability issue**; for deep neural networks with 50+ layers, **manually tuning these scales is impossible**. If the scale is slightly off, the activations will either explode to infinity or vanish to zero as they propagate through the layers.
- The **Goal**: We need a **principled, mathematical** way to initialize weights so that activations remain well-behaved (roughly gaussian with unit variance) throughout the entire network, without requiring manual tuning.

#### Variance Addition
Before talking about why we need to **preserving Variance**, i need to clarify the mathematical Rule (Variance Addition):
When a neuron computes a dot product, it performs a weighted sum of all its inputs: $$Sum = (x_1*w_1)+(x_2 * w_2)+...+(x_n*w_n)$$if we assume each individual connection $(x_i *w_i)$ acts as an independent random variable with its own variance, the total variance of the sum is the addition of each individual variance: $$\text{Var}(\text{Sum}) = \text{Var}(x_1 \cdot w_1) + \text{Var}(x_2 \cdot w_2) + \dots + \text{Var}(x_n \cdot w_n)$$
If every connection has the exact same variance (let's call it V), then: $$Var(Sum) = n*V$$**The total variance scales linearly with the number of inputs.** If you have 30 inputs, the variance of the output is 30 times larger than the variance of a single input.

#### Preserving Variance
Andrej demonstrates the math using a separate code snippet to simulate a layer:
- **Variance growth**: if you have **input x** and **weights w** both initialized with standard gaussian distribution (variance = 1), their matrix multiplication results in an output with a much larger variance. For example, with 10 inputs, the standard deviation grows from 1.0 to approx 3.0. As we can see in the code:
```python
import torch
import matplotlib.pyplot as plt

# 1000 example, 10 dimensional
x = torch.randn(1000, 10) / 10 ** 0.5 # ** 0.5 =convert the square root to exponent, 10 = fan_in
#fan_in = number of input connection

# A layer with 200 neuron that takes 10 dimensional input
w = torch.randn(10, 200)

# preactivation
y = x @ w

print(x.mean(), x.std())
print(y.mean(), y.std())

plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True);
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True);
plt.show() # Added to display the plot
```
![[Pasted image 20261104015109.png]]
- We want the **activation throughout a deep neural network to be roughly Unit Gaussian** (Mean ≈ 0, Variance ≈ 1\) **because it is the only way to keep the signal “alive” and “learnable”** as it passes through many layers.  If the distribution drifts away from this "Unit Gaussian" sweet spot, you run into two catastrophic failures: **Vanishing** or **Exploding**.  
	- **Vanishing**: If the variance of your activations drops below 1.0 (e.g., to 0.5) at each layer: The signal shrinks → 0.5\*0.5\*0.5 … rapidly approaches zero. By the time the signal reaches the 50th layer, the activations are all effectively zero. Why it’s bad?  
		* **Forward Pass:** The network output becomes constant (zero) regardless of the input image/text. It carries no information.  
		* **Backward Pass:** Gradients are calculated based on the activations. If activations are zero, the gradients become zero. The network stops learning completely.  
	- **Exploding**:  If the variance grows larger than 1.0 (e.g., to 1.5) at each layer; The signal amplifies. $1.5*1.5*1.5$ … grows exponentially. By the deep layers, the numbers become massive (millions, billions, then NaN). Why it's bad?  
		* **Forward Pass:** The computer cannot handle these numbers (floating point overflow).  
		* **Saturation:** Even before crashing, these huge numbers will push every Tanh/Sigmoid neuron instantly into the "saturated" zone (outputting fixed \-1 or \+1), causing the gradient to die immediately.

**The fix (Fan-in normalization)**: To prevent this expansion and keep the output variance equal to 1.0, you must scale the weights by dividing by the square root of the “fan-in” (**the number of input connections**). Mathematically: $$W~ Scale = \frac{1}{\sqrt{fan-in}}$$
- **Visualizing “fan-in”**: imagine a signal neuron in a hidden layer. It doesn’t just sit there; it is **connected to the previous layer**.
	- If the previous layer has 100 neurons, this single neuron has 100 coming wires. Each wire carries a signal (an input value). Therefore, the **fan\_in is 100**.  
	* “Input connections” is just a count of how many numbers are being summed up inside the neuron. In the video’s case, it is 30\.
	* ![[Pasted image 20261104020008.png|294]]
- **Example**: In Andrej’s makemore example, he calculates the fan-in specifically for his hidden layer:
	- **The Context:** The model looks at **3** previous characters to predict the next one.  
	* **The Embedding:** Each of these characters is converted into a vector of **10** numbers (the embedding size).  
	* **The Input Vector:** These are flattened into a single long list of inputs. 3 Characters * 10 numbers/char = 30 total inputs.  
	* The Fan\_in: Therefore, every neuron in the hidden layer is receiving 30 input connections fan_in = 30.
* Why **fan_in** Causes Variance to Grow: The reason we divide by the square root of the fan-in is statistical.
	* **The Operation:** A neuron performs a dot product: it sums up (input * weight) for all its connections.  (Variance Addition)
	* **The Problem:** If you add up two random numbers, the result has some variance (spread). If you add up **30** random numbers, the result has roughly **30 times** the variance. The more inputs you have, the more the sum "explodes" away from zero.  
	* **The Solution:** To keep the output stable (variance ≈ 1.0), you must scale the weights down. The more inputs (fan-in) you have, the smaller the weights need to be to compensate.
#### Kaiming initialization
“**Delving Deep into Rectifiers**” by He et al (2015), which formalized this modern activation functions.
- **The ReLU adjustment**: The paper specifically analyzed **ReLU** (Rectified Linear Unit) activations. Since ReLU sets all negative numbers to zero, it **"throws away" half the distribution**. To compensate for this loss of variance, you need to multiply the variance by 2. Kaiming Formula for ReLU: $$W~ Scale = \frac{2}{\sqrt{fan-in}}$$
- **Forward vs. Backward Pass**: The paper proves that if you initialize properly to preserve variance in the forward pass, the gradients in the backward pass are also approximately preserved.

**Adapting for TanH (The “Gain”)**  
Since in the video we use Tanh instead of ReLU, the formula needs a slight adjustment.
* **What is “Gain**”? Think of a neural network layer as a sequence of two operations  
	1. **Matrix Multiplication**: This creates the pre-activation ($z$). If we initialize weights properly (using standard fan-in formula), this step **preserves the signal strength** (variance ≈ 1\)  
	2. **Activation function**: This applies non-linearity (like Tanh or ReLU) to $z$.  
* **The Problem:** Most activation functions change the "loudness" (variance) of the signal.  
	- **Gain** is the number you multiply the weights by to cancel out this change.  
	- **Goal:** Input Variance* Weight Scaling* Activation Effect ≈  Output Variance (Keep it steady at 1.0).  
* “**Gain”** and **initialization** are combine into a single number that represents the “proper initialization” for a specific layer.  
* **Why Tanh needs a boost:** Tanh is "contractive"—it squashes the tails of the distribution towards \-1 and 1\. To fight this squashing and keep the data spread out, the weights need to be boosted by this$\frac{5}{3}$ factor.  
* “Gain” for different function:   
	- Linear/identity: Gain = 1
	- ReLU:  Gain = $\sqrt{2}$ --> square root multiplication rule, the W scale of ReLU is unchanged.
	- Tanh : Gain = $\frac{5}{3}$（approx 1.67)
Andrej replaces the manual `0.2` scale with the calculated Kaiming initialization:
- **Calculation**:  
	- Fan-in: 30 (3 characters * 10 dimensions each)  
	- Gain: $\frac{5}{3}$  (for Tanh)
	- The Formula: Scale = $\frac{5/3}{\sqrt{30}}$ = 0.3.
- **Result**: He updates the code to multiply the random weights by roughly 0.3 instead of 0.2
- **Outcome:** The network trains just as well (validation loss ~2.10), but the initialization is now **principled** and robust enough to work on much deeper networks without manual fiddling.
The Gain applies only once, before the training loop starts. ==If you applied it every epoch, your network would explode immediately==.
- **Initialization (The `5/3` Gain):** This is like positioning a runner at the starting line. You want them in a fair position so they don't trip immediately. We set the weights so that the signal passes through the network clearly *at the very first moment*.  
- **Training (Optimization):** Once the race starts, **Gradient Descent (SGD)** takes over. SGD looks at the loss and decides: "Hey, the signal is getting too small here (massive loss), I should increase the weights," or "The signal is exploding (tanh saturates), I should decrease them."
Tanh “squash” every time, but we only need to manually interfere at step 0\. Because **at Step 0:** The weights are random. They don't know they are part of a deep network. They don't know they need to be larger to fight the Tanh. So we force them (multiply by 5/3). **At Step 1 to 1000:** The weights are now "smart." Through backpropagation, the gradients tell the weights: *"If you get smaller, the signal dies and the loss goes up."* Because the optimizer is trying to lower the loss, it will automatically prevent the weights from shrinking too much. The network **learns to maintain the variance** itself.

**Layer-Specific Scaling:** Kaiming initialization calculates the scaling number for the weights based on the number of input units (fan-in) to _that specific layer_. Because different layers for example in a 50-layer network (like a ResNet-50) have different dimensions and channel sizes, each layer requires its own uniquely scaled initialization.
```python
# Standard PyTorch call for He Initialization  
torch.nn.init.kaiming_normal_(  
    tensor,  
    a=0, # Default
 # if a = 0, this means you are using the standard ReLU. The negative slope is 0. if a>0, this assumes you are using Leaky ReLU (or PReLU)  
    mode='fan_in',    # <--- The default is to look at Inputs  
    nonlinearity='relu' # used to calculate the “Gain”  
)
```
The manual weight initialization in nows days are less critical compared to 2016, because of the modern innovations → **Normalization layers** (Specifically **Batch Normalization**), followed by architectural improvements like **Residual Connections** and better Optimizers that we are going to cover.
How we initialize it with plain code:
```python
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)  * (5/3)/(n_embd*block_size) **0.5
```
#### **Note: We don't proper initialize the last layer**
When we talk about “proper initialization” ( like Kaiming init) for hidden layer, the goal is to preserve variance (keep the signal strength at 1.0). BUt for the last layer, the goal is different: we want to **destroy variance** (squash the signal to 0).  That’s why Andrej scaled down the weights of the last layer (multiplied by 0,1) instead of “proper” kammining initialization or leaving them random:

**Standard/Kaiming Init is "Too Loud" for the Last Layer:** If you use standard random initialization (or Kaiming) on the last layer, the weights will be roughly between \-1 and 1\.
* **Result:** The output logits will be numbers like `[-1.5, 0.2, 2.3, -0.8]`.  
* **The Softmax Problem:** Softmax exaggerates differences. Even small differences (like 2.3 vs \-0.8) turn into massive probability differences (e.g., 80% vs 1%).  
* **The Consequence:** The network starts off "confidently wrong." It thinks the next character is "C" with 80% confidence, but it's actually "A". This causes the massive initial loss (27.0).

**Scaling Down "Silences" the Last Layer:** By multiplying the weights by 0.01, Andrej forces the weights to be tiny (e.g., 0.005).
* **Result:** The output logits become effectively zero: \[0.001, \-0.001, 0.000, 0.002\].  
* **The Softmax Benefit:** When you feed zeros into Softmax, it calculates e^0 \= 1\. Since all logits are equal, the probabilities become perfectly uniform (1/27 for everyone).  
* **The Consequence:** The network starts off "ignorant but humble." It admits it doesn't know the answer and guesses everything equally. This gives the optimal starting loss (3.3).

Example: How should we proper initialize the weights for a simple Model:
```python
import torch
import torch.nn as nn

class TanhMLP(nn.Module):
    def __init__(self, input_size=30, hidden_size=200, num_classes=27):
        super().__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        # Hidden layer followed by tanh
        tanh_gain = nn.init.calculate_gain("tanh")  # 5/3
        nn.init.xavier_normal_(
            self.hidden.weight,
            gain=tanh_gain,
        )
        nn.init.zeros_(self.hidden.bias)

        # Small final weights produce nearly equal initial logits.
        nn.init.normal_(self.output.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        return self.output(x)

model = TanhMLP()
```

### Batch Normalization
Batch Normalization (2015) is a critical innovation that stabilized the training of deep neural networks, **moving us away from the fragile “balancing act”** of weight initialization.
We want the activations (hidden states) throughout the network to be roughly Gaussian (mean 0, variance 1). This keeps the `tanh` nonlinearities in their active "linear" region and prevents gradients from vanishing or exploding. Instead of carefully tuning weight initialization to *hope* the activations come out Gaussian, **Batch Normalization** simply calculates the mean and variance of the current batch and **forces** the data to be Unit Gaussian.  **Calculating the mean** and **dividing by the standard deviation** is a **perfectly differentiable operation, meaning we can backpropagate through it**.
![[Pasted image 20261106014829.png]]
#### **The implementation**
1. **Calculate Batch Statistics** for a hidden layer with inputs `h_preact` (shape: 32 examples * 200 neurons):  
	1. Mean: Calculate the average activation for each neuron across the batch  
	2. Variance/Std: Calculate the the spread (standard deviation) for each neuron across the batch.
2. **Normalize**: applies the normalization formula: $\hat{x} = \frac{x - \text{mean}}{\sqrt{\text{variance} + \epsilon}}$
	1. Now, every neuron’s firing rate is exactly Unit Gaussian for that batch. `hpreact = (hpreact - hpreact.mean(0,keepdim=True)) / hpreact.std(0,keepdim=True)`
	2. ![[Pasted image 20261106015440.png]]
	3. Visualization:  In the picture below, we can see the effect of normalizing data. The original values (in blue) are now centered around zero (in red). This ensures all the feature values are now on the same scale.![[Pasted image 20261106015611.png]]
	4. **What happens if we do not normalize?** As we talked before; Since the network output is a linear combination of each feature vector, this means that the network learns weights for each feature that are also on **different scales**. if we do not normalize, the **large feature will simply drown out the small feature.** Then, during gradient descent, in order to "move the needle" for the Loss, the network would have to make a large update to one weight compared to the other weight. This can cause the gradient descent trajectory to **oscillate back and forth** along one dimension, thus taking more steps to reach the minimum. ![[Pasted image 20261111042540.png|291]]
3. **Scale and Shift (γ and β)**  
	* **Problem:** Forcing activations to be perfectly Gaussian might be too restrictive. Sometimes the network *wants* to saturate a neuron (e.g., to output -1 or 1).  
	* **Solution:** Introduce two learnable parameters per neuron:  
		- **γ (Gamma/Gain):** A scaling factor (initializyed to 1). 
			- `bngain = torch.ones((1,n_hidden))`
		- **β (Beta/Bias):** A shifting factor (initialized to 0).
			- `bnbias = torch.zeros((1,n_hidden))`
		- Final Output: $y=\hat{x} \cdot \gamma + \beta$
			- `hpreact = bngain * (hpreact -hpreact.mean(0,keepdim=True)) / hpreact.std(0,keepdim=True) + bnbias`
		- This allows the network to “undo” the normalization if it wants to. It starts at Unit Gaussian (Stable initialization) but can learn to shift the distribution wherever it needs during training.
There is a weird side effect: **Mathematical Coupling**.
- In a standard network, one image’s output depends only on that image.  
- In BatchNorm, one image's output depends on **the other images in the batch** (because they **influence the mean/std used for normalization**). Batch Normalization breaks this independence. Because it introduces a step where the calculation for Image A relies on statistics derived from **Image A \+ Image b \+ Image C…**
	- **The Math:** To normalize image A, the network subtracts the Batch Mean (μ). 
		- $\text{Normalized A} = \frac{A - \mu}{\sigma}$
	- **The problem**: The variable $\mu$ is the average of everything in the current batch.
		- If the batch is `[Image A, Image B]`, the mean is $\frac{A+B}{2}$
		- If the batch is `[Image A, Image C]`, the mean is  $\frac{A+C}{2}$
	- **The consequence**: The exact numbers flowing through the network for Image A change depending on who its “neighbors” are in the batch.  
	* **Why it’s “terrible”**: This is conceptually messy. A function f(x) should ideally just return the result for x. Instead, f(x) now returns “a result for x, assuming it is hanging out with y and z”.
* **Jitter/Noise:** Depending on which random images are in your batch, the output for a specific image "jitters" slightly. Because you shuffle your training data before every epoch, Image **A will be paired with different random images every time** it is seen.  
	* **Epoch 1:** Batch is `[Dog, Cat, Truck]`. The "Dog" is normalized against the mean of this group.  
	* **Epoch 2:** Batch is `[Dog, Airplane, Ship]`. The "Dog" is normalized against a completely different mean.  
	* **The Effect:** The neural network sees the "Dog" image slightly differently every time. The features wobble or "jitter" slightly. It's like looking at the same photo through slightly different colored sunglasses every time.   
- **The "Regularization" Benefit:** Surprisingly, this noise acts as a regularizer (similar to Data Augmentation), making it **harder for the network to overfit**. This beneficial side effect is one reason BatchNorm is still used today despite its complexity.
**It’s customary to take a linear layer or a convolutional layer and append a batch normalization layer right after it. To control the scale of activations at every point in the neural network.**

**The Inference Problem (Test Time):** A major issue arises at test time: **"How do we predict for a single image?"**
- You cannot calculate a "batch mean" from a single input.
- **The Solution (Running Mean/Std):**  
	  * During training, we maintain a separate "running mean" and "running variance" alongside the backpropagation.  
	  * These are updated using a "momentum" strategy (e.g., `0.999 * old + 0.001 * current`).  
	  * These running stats are **not trained with gradient descent**; they just track the moving average of the data distribution.  
* **Result:** At test time, we ignore the batch statistics and use these **pre-calculated "running" statistics to normalize the single image**.
```PYTHON
 # Batch Normalization
bnmeani = hpreact.mean(0, keepdim = True)
    
bnstdi = hpreact.std(0,keepdim=True)
    
hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    
with torch.no_grad():
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi**
```
**The Redundant Bias Term**, Andrej points out a common efficiency tip:
- if you have a linear layer followed by BatchNorm: $$\begin{aligned}
\text{Linear: } & Wx + b \\
\text{BatchNorm: } & \frac{(Wx + b) - \text{mean}}{\dots}
\end{aligned}$$
- The batch normalization step subtracts the mean. Since the bias ***b*** is a constant added to every example, it effectively becomes part of the mean. **The subtraction cancels it out completely**.  
- **Conclusion**: The bias in the Linear layer is useless ("spurious") if it is followed by BatchNorm. You should set bias=False in your Linear/Conv layers and let the BatchNorm's **$\mu$** parameter handle the shifting.

### ResNet-50 (Residual Network) walk through
Andrej opens the PyTorch source code (or a standard implementation) for the **Bottleneck Block**, which is the fundamental building block of ResNet50. He highlights the repeating "Motif" of the layers:
* **The Pattern:** Conv2d → BatchNorm2d → ReLU.  
* **Repetition:** This sequence is repeated three times inside a single block (1x1 conv, 3x3 conv, 1x1 conv).  
* **Residual Connection:** At the end of the block, the input x is added to the output (x += out), which is the signature "skip connection" of ResNets.
He points out a specific argument in the `Conv2d` layer definition: **`bias=False`**. In the code, the convolution layers are initialized with `bias=False`. just as he demonstrated earlier with the Linear layer, if a Convolution layer is immediately followed by a Batch Normalization layer, the bias of the Convolution is **useless**. The **BatchNorm layer subtracts the mean**, which cancels out any bias added by the convolution.  The BatchNorm layer has its own "bias" (parameter β) that handles the shifting. This validates his earlier advice to remove biases from layers preceding BatchNorm to save parameters and memory.

**`torch.nn.BatchNorm1d`** He compares his `BatchNorm1d` class to the official PyTorch version, noting the arguments match almost perfectly. `torch.nn.BatchNorm1d(_num_features_, _eps=1e-05_, _momentum=0.1_, _affine=True_, _track_running_stats=True_, _device=None_, _dtype=None_, _*_, _bias=True_):
* **`eps` (epsilon):** A small value (default `1e-5`) added to the variance to prevent division by zero.  
* **`momentum`:** Used for the running mean/variance calculation.
	* ```python
	  bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
	  bnstd_running = 0.999 * bnstd_running + 0.001 * bnstd
	  ```
	* PyTorch defaults to `0.1`, whereas Andrej used `0.001`. He explains this depends on batch size; standard ImageNet training uses large batches so `0.1` is fine, but for small batches, a smaller momentum is smoother.
	* To decide the correct momentum, you have to ask: “How noisy is my current batch?”.   
		* **Case A**: If you average 256 images, the resulting mean is statistically very stable and accurate. It is a good representation of the "true" dataset. You can **trust** this batch. Use a higher value (like **0.1**). You can afford to update your running stats quickly because the new data is reliable.  
		* Case B: If you average only 32 examples, the result is noisy. One weird example can throw off the average significantly. You **cannot trust** this single batch. It is too "jittery.". Use a very low value (like **0.001**). You want to ignore the noise of the specific batch and only shift the running average very slowly. This forces the network to look at thousands of batches before significantly changing its mind about the mean.
* **affine=True:** This enables the learnable parameters γ (gamma) and β (beta). $y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta$.
* **`track_running_stats=True`:** This tells the layer to maintain the "running mean" and "running variance" buffers (non-learnable) to be used during inference (testing).
### **Pytorch-ifying the code**
Previously, the code was a script with global variables (W1, b1, W2, b2, ..) and manual formulas. This is hard to scale to deeper networks. Karpathy wants to create classes that act as self-contained "layers" so they can be stacked in a list, exactly how `torch.nn.Sequential` works.  
Andrej implements three key classes, designing their API to match PyTorch (`__init__` for setup, `__call__` for the forward pass, and `parameters()` for optimization).
```python
class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out

  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias]) #[self.weight, self.bias]
```


```pYTHON
class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True

    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  

  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance

    self.out = self.gamma * xhat + self.beta
    
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out

  

  def parameters(self):
    return [self.gamma, self.beta]
```

Instead of hardcoding layers, he creates a list called `layers` that defines the entire architecture in order.
```PYTHON
layers = [
  Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),

  Linear(           n_hidden, vocab_size, bias=False), BatchNorm1d(vocab_size),
]
```
### **Forward Pass activation statistics**
Andrej visualizes the histograms of the Tanh outputs (Activations) for every layer in the network.
- **The Metric:** He calculates the **Standard Deviation** and the **Saturation Percentage** (how many neurons are outputting values very close to \-1 or 1, specifically `abs(h) > 0.97`).  
- **The "Sweet Spot" (Gain \= 5/3):**  
	* When using the theoretical optimal gain for Tanh (5/3 ≈ 1.67), the histograms look consistent across all layers.  
	* **Result:** The standard deviation stays stable at around **0.65** and saturation is manageable at **\~5%**. This means the signal flows deeply through the network without dying out or exploding.  
		- 0.65 appears naturally when you feed a Unit Guassian into a Tanh function.   
			* **Step 1 (BatchNorm):** The Batch Normalization layer forces the input (pre-activation) to be a Unit Gaussian distribution. Input Variance ≈1.0 (Standard Deviation ≈ 1.0).  
			* **Step 2 (Tanh Squashing):** When you pass this distribution through `tanh(x)`:Values near 0 (e.g., 0.1) stay roughly the same. Values near the edges (e.g., 1.5) get squashed down (to \~0.9).  
			* **The Result:** Because `tanh` squashes the large numbers, the "spread" (Standard Deviation) of the data shrinks.  
			* Mathematically: **Std(Tanh(Unit Gaussian))** ≈ **0.63 \- 0.65.**
	* ![[Pasted image 20261113022158.png]]
	* ~5% saturation is good because **It proves you are using the non-linearity:** Some neurons are being pushed into the curved tails. This provides the necessary "twisting" and "bending" of the data manifold that allows Deep Learning to solve complex problems. 
		* **It preserves the gradient:** 95% of neurons are still active (providing flow), while 5% provide the necessary structure/curvature.
* **The Fail Case: Low Gain (e.g., 1.0):**  
	* If the gain is set to 1.0 (treating Tanh as if it were linear), the Tanh function **"squashes" the data slightly at every layer**. So without gain to keep expanding it ot fight the squashing, the result we got is:  
	* **Result:** The histograms get narrower and narrower. By the deeper layers, the standard deviation shrinks toward zero. The signal effectively vanishes.
	* ![[Pasted image 20261113023628.png]]
	* 0% saturation means all your data is sitting comfortably in the center, linear region of Tanh curve.
		* Mathematically in this region, tanh(x) \= x.  (when x = 0, center region, tanh(x) = x)
		- **Consequence:** Your Deep Neural Network effectively becomes just a stack of Linear Layers. **y = W_3(W_2(W_1(x)))**. Because matrix multiplication is associative, this collapses into a single matrix multiplication. Result → You have built a "deep" network that has the learning capacity of a single shallow layer. It cannot learn complex shapes, curves, or decision boundaries. It can only draw straight lines.
- **The Fail Case: High Gain (e.g., 3.0):**  
	* If the gain is too high, the weights expand the data too much.  
	* **Result:** The Tanh neurons get pushed into their flat tails. The saturation spikes to nearly **100%**, causing a "binary" state where neurons are stuck at \-1 or 1, killing the gradients.
	* ![[Pasted image 20261113023926.png]]
### **Backward Pass Gradient Statistics**

Andrej performs a similar visualization but for the **Gradients** (`grad`) flowing backward during backpropagation. 
* **The Goal:** We want the gradients to have roughly the **same magnitude** (Standard Deviation) across all layers. If gradients in early layers are vastly different from later layers, the network will train unevenly. You want **standard deviation and mean of gradient remain roughly constant across all layers**. **if you see the gradient dropping (e.g., Layer 4 is 1e-2, layer 0 is 1e-5) you have the vanishing gradient problem**.  
* **The Observation:**  
	- With the correct gain (5/3), the gradient histograms are aligned (stacked on top of each other), meaning all layers are learning at roughly the same speed.
		- ![[Pasted image 20261114014846.png]]
	- With **Low Gain (0.5):** He shows that while activations shrink, the gradients actually **explode** (expand) as they travel backward.
		- ![[Pasted image 20261114014926.png]]
	- With **High Gain (3.0):** The gradients exhibit asymmetry and instability, with deeper layers having different gradient scales than earlier layers.
		- ![[Pasted image 20261114014956.png]]
To prove the math, he creates a "Linear Sandwich" by removing all `Tanh` layers, leaving only `Linear` layers stacked on each other.
* **The Collapse of Gain:**  
	- He shows that if you keep the Tanh gain (5/3) in a purely linear network, the activations **explode** to infinity (because 1.67 > 1.0, so the signal amplifies at every layer).  
	- He sets the gain to **1.0** (the correct gain for linear/identity). Now, the forward and backward passes are perfectly stable.  
* **Why we need Non-Linearities:**  
	- He explains that while a deep linear network is easy to initialize (Gain = 1.0), it is useless.
	- **Mathematical Collapse**: A stack of 10 Linear layers ($W_{10}*W_{9}*...*W_{1}*x$) is mathematically equivalent to a **Single Linear Layer** ($W_{final}*x$).
	- The `Tanh` layers are essential because they prevent this collapse, giving the network the "complexity" to approximate arbitrary functions, even though they introduce the difficulty of tuning the "Gain."
```python
# visualize histograms
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []

# skip the last because the last layer is usually the loss calculation or raw logits
# which behaves diffrently. We are most interested in the "hidden" layers where the signal tensd to get lost
for i, layer in enumerate(layers[:-1]): # note: exclude the output layer

  # only look at the activation layers
  # The activation function is usually the "bottleneck" where gradients die. If the Tanh neurons are "saturated" (outputting -1 or 1),
  # the gradients becomes 0, effectively stopping the learning process for that neuron

  if isinstance(layer, Tanh):
    # out stored output from the forward pass
    t = layer.out.grad
    # you want stand deviation and mean of gradient remain roughly constant across all layers.
    # if you see the gradient groping (e.g., layer 4 is 1e-2, layer 0 is 1e-5) you have the vanishing gradient problem.
    # The earlier layers aren't learning anything
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))

    # histogram() takes in the tensor t and sort them into "bin"
    # density = True:
    # False(default): The y-axis would show the count (E.g., "500 values are near zero")
    # True: The y-axis shows probability density. This is crucial because different layers have different numbers of parameters.
    # Layer 1 might have 1,000 parameters and Layer 2 might have 10,000. Density = True normalizes them so their curves are comparable on
    # the same plot, regardless of size
    hy, hx = torch.histogram(t, density=True)

    # length of hy = 100, lenght of hx = 101
    # because if you have to build 3 sections of fence, you need 4 posts.
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__}')

plt.legend(legends);
plt.title('gradient distribution')
```

### **Parameter activation & Gradient statistics**
Andrej moves beyond just looking at activations (outputs of layers) to looking at the weights (parameters) themselves.
* **What is being visualized?** He iterates over all learnable parameters (specifically the weight matrices of the Linear layers, ignoring biases for simplicity) and plots:
	1. **Mean & Standard Deviation of Weights:** To check if the weights themselves are distributed well (e.g., are they all zero? are they exploding?).  
	2. **Gradient-to-Data Ratio:** This compares the magnitude (std dev) of the **gradients** (`p.grad`) to the magnitude of the **weights** (`p.data`).
		- The ratio give us the relative scale or context of the **signal**. A gradient of 0.1 could be “tiny” or “massive”, depending entirely on how big the weight (`p.data`) is. By calculating $\frac{std(gradient)}{std(weights)}$, we normalize the gradient against the size of parameter. It tells us **how "violent" the gradient signal is for that specific layer.**
					1. **Case A:** If the weight is **1000.0**, a change of 1.0 is a rounding error (0.1% change). The weight is stable.
					2. **Case B:** If the weight is **0.01**, a change of 1.0 is catastrophic (10,000% change). The weight effectively explodes or flips sign instantly.
		* In a deep network, you want all layers to learn roughly at the same speed. If Layer 1 has a ratio of `0.001` and Layer 5 has a ratio of `10.0`, Layer 5 is "thrashing" (changing wildly) while Layer 1 is frozen.  
		- Ultimately, we use gradients to update weights: **New Weight \= Old Weight \- ( Learning Rate \* Gradient)**. The Gradient-to-Data ratio helps you estimate what the Learning Rate should be.
			- If your Gradient-to-Data ratio is huge (e.g., 100), you know you need a **very small** learning rate to prevent the weights from exploding.  
			* If the ratio is tiny (e.g., 1e-5), you know you need a **large** learning rate to get the weights to move at all.
* **The "Troublemaker" Last Layer:**  
	- **Observation:** Most layers show consistent statistics. However, the **last layer** (plotted in pink) is a distinct outlier.  
	- **Reason:** Recall that he manually shrunk the weights of the last layer (multiplied by 0.1) to ensure a uniform probability distribution at initialization. Because the weights are tiny, but the gradients are normal sized, the **Gradient-to-Data ratio** for the last layer is huge (about 10x larger than other layers).  
	- **Consequence:** This means the last layer is training much faster than the rest of the network initially. While not fatal, it's a sign of a "miscalibration" that simple SGD has to work through.
### **Update-to-Data Ratio (ud)**
Andrej argues that looking at the raw gradients (Gradient-to-Data) isn’t actually the most useful metric because the “Step size” depends on the Learning Rate. He introduces a more powerful metric: The **update-to-Data Ratio**.
- The Formula: $Ratio =\frac{std(update)}{std(data)} = \frac{std(learningRate \times Gradient)}{std(Weights)}$
- **What it tells you:** It answers: *"How big is the step we are taking relative to the size of the parameter?"*
	- If a weight is roughly `1.0` and the update is `0.001`, the ratio is `1e-3`.
- **The "Golden Rule" (Heuristic):** Karpathy shares a rough "rule of thumb" that he uses in practice: **You want this ratio to be around $10^{-3}$ (or -3.0 on the log plot).**
	- This means your updates should be roughly **1/1000th** of the parameter's value. Example, if a weight is 1.0, you update it to 1.001 or 0.999  
	* **Why not lower?**  If you are far below the line (e.g., -5.0), you are changing the weights by tiny, microscopic amounts $10^{-5}$. It’s too slow, you are wasting time and electricity.  
	* **Why not higher?** if you are high above the line (e.g., -1.0), you are changing the weights by massive amounts ($10^{-1}$ or 10%). It’s **unstable**. Neural network training relies on the assumption that the "Loss Landscape" is locally **smooth**. If you take too big a step, that assumption breaks, and the **gradient becomes inaccurate**.
	* $10^{-3}$ strikes the perfect balance:
		* **Fast enough:** Over 1,000 steps, these 0.1% changes compound significantly, allowing the network to reshape itself and learn the data patterns quickly.  
		- **Slow enough:** The changes are small enough that the training remains smooth and stable. The network doesn't "forget" everything it learned in the previous step.
![[Pasted image 20261114024545.png]]
- **Diagnosing Learning Rates:** He demonstrates how this plot instantly reveals if your Learning Rate is good or bad:  
	* **Too Low (e.g., -5.0):** If the curves are far below the -3.0 line, your updates are tiny. The network is learning, but **way too slowly**. You should increase the Learning Rate.  
	* **Too High (e.g., -1.0):** If the curves are high up, your updates are massive (e.g., changing the weights by 10% or more every step). This risks "thrashing" or exploding.  
	* **Just Right (~ -3.0):** The curves should hover slightly above or around the \-3.0 line.  
- **Behavior Over Time:** You can see the curves evolve. The last layer starts very high (large relative updates because weights are small) but eventually settles down as the weights grow and the gradients stabilize. This visualization is a powerful tool for debugging "invisible" training issues where the loss goes down, but layers are training at vastly different speeds.

### Bring back BatchNorm
Andrej inserts the BatchNorm1d layers back into the layers list.
- **Placement:** He places them **after** the `Linear` layers but **before** the `Tanh` non-linearities.  
	* *Note:* He mentions that placing them after the non-linearity is also possible and often works fine, but the standard practice (and his implementation) is `Linear -> BatchNorm -> Tanh`.  
- **Expectation:** Since BatchNorm explicitly forces the output of every layer to be Unit Gaussian (before the learnable scale/shift), he expects the activation statistics to look perfect regardless of what happened in the Linear layer before it.

When he runs the training with BatchNorm, the diagnostic plots look ideal immediately:
- **Activations (Viz \#1):**  
	* The histograms are consistent ("homogeneous") across all layers.  
	* The standard deviation stabilizes at roughly **0.65**.  
	* The saturation stays at a healthy **\~2-5%**.  
	* **Why:** BatchNorm mathematically guarantees this distribution at initialization, so the network doesn't have to "learn" to be well-behaved.  
- **Gradients (Viz \#2):** The gradients are stable and do not vanish or explode.  
- **Update Ratio (Viz \#4):** The update-to-data ratio is consistent across layers, hovering slightly above the `-3.0` (log scale) "sweet spot."

To prove that BatchNorm is doing the heavy lifting, Karpathy intentionally "breaks" the initialization of the Linear layers to see if BatchNorm can fix it.

**Experiment A: Terrible Gain (0.2)**
* **The Change:** He changes the initialization gain of the Linear layers to **0.2** (instead of the optimal 5/3). In a standard network, this would cause the signal to vanish immediately.  
* **The Result:**  
	  * **Activations:** They look **exactly the same** (perfect). BatchNorm sees the tiny signal, divides it by its tiny standard deviation, and boosts it back up to Unit Gaussian.  
	  * **Update Ratio:** However, the *update ratio* (learning speed) drops. This shows that while the forward pass is fixed, the effective learning rate might need tuning.

**Experiment B: Removing Fan-in Scaling**
* **The Change:** He completely removes the 1/sqrt(fan\_in) scaling, initializing weights with a simple Standard Gaussian (variance = 1.0). In a normal network, this would cause massive explosion of values.  
* **The Result:**  
	  * **Activations:** Still look good. BatchNorm suppresses the explosion.  
	  * **The Problem:** The **Update-to-Data Ratio** drops significantly below -3.0 (e.g., to -4.0). This means the network is training **way too slowly**.  
	  * **The Fix:** He demonstrates that you can simply **increase the Learning Rate** (e.g., by 10x) to get the ratio back to -3.0.