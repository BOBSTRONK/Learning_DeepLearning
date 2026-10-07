### Intuitive definition of a derivative
Rather than relying solely on symbolic calculus rules (like the power rule), We focuses on the mathematical definition of a derivative:$$\lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
The **meaning** of the formula is the derivative measures the **sensitivity of the function to its input**. It answers the question: if you **if you bump the input $x$ by a tiny amount of $h$. How does the function's output respond?** Does it go up or go down, and by how much? This response is effectively the **SLOPE**of the function at that specific point. 
- The **SLOPE** of a function is a measure of its **steepness** and the **direction** it is moving. Think of it as the "rate of change". 
- $f(x+h)$ represents the function's new value after input $x$ changed
- $f(x)$ represents the function's value at the original point.
- $f(x+h) - f(x)$ represents exactly how much the output moved in response to the nudge (small value, 中文：轻推).
- Dividing by ***h*** is the crucial step that turns a simple “difference” into a **rate of change** (or slope). **How outputs changes with respect to changes in input**. 
	- Example: 
		- Let's look at the example when the slope was $3$:
			- **Step size (h)**, change in input = 0.001  
			- **Change in the output**（$f(x+h) - f(x)$ ) = 0.003
		- If we didn't divide by ***h***, we would just say "the function changed by 0.003." That sounds like a tiny change! It implies the function is essentially flat. But when we normalize by the input step: $\frac{0.003 \text{ (change in output)}}{0.001 \text{ (change in input)}} = 3$, Suddenly, we see the truth: for every 1 unit you move right, the function goes up 3 units. That is actually quite steep\! The division reveals the **intrinsic sensitivity** of the function, regardless of how tiny your test nudge (***h***) was.
	- **We want “Steepness”, not just “Distance”.**  By dividing by ***h***, you are calculating "how much do I go up **per single step** forward?" This standardizes (normalizes) the measurement so it represents the **angle** of the slope rather than just the distance traveled.
		- Imagine you are hiking up a mountain.  
			- **Numerator ( $f(x+h) - f(x)$ )**: This is how many meters you climbed up (**Rise**).  
			- **Denominator ($h$)**: This is how many meters you walked forward (**Run**)
		- If you only looked at the numerator (the rise), you wouldn't know how steep the hill is.
			- *Scenario A:* You climbed 10 meters up. (Did you walk 10 meters forward? That's a 45-degree angle. Very steep\!)  
			- *Scenario B:* You climbed 10 meters up. (Did you walk 10 kilometers forward? That's practically flat.)
**Demonstration:** How to calculate the derivative numerically using python, which is core to understand how neural networks work "under the hood":
- **Function**: $f(x) = 3x^2 - 4x + 5$ 
	```python
	def f(x):
		return 3*x**2 - 4*x + 5
	```
- Visualization: 
	```python
	xs = np.arange(-5, 5, 0.25)

	ys = f(xs)

	plt.plot(xs, ys)
	
	```
![[Pasted image 20261002023604.png]]
- **Step size (h)**: He chooses a very small number for ***h*** (e.g., 0.001) to approximate the limit as ***h*** goes to zero.
- - **Example 1 (x \= 3.0):**  
	* He calculates the function at **$x = 3.0$** and finds **$f(3) = 20$**.  
	* He adds the small nudge **$h$** to getting $f(3+h)$. Since the slope is positive at this point, the result is slightly larger than 20.  
		- **Positive Slope**: The line goes up as you move left to right.  
	* The numerical slope is approximately 14. We can verifies this mentally using calculus $f’(x) = 6x - 4 → 6(3) - 4 =14)$, confirming the numerical method works.  
		* $f'(x)$ is the derivative, derivative of a function represents the **instantaneous rate of change**.  A positive slope means the function is going up as you moving from left to right (or as value $x$ increases)
- **Example 2( X \= \-3.0):**  
	* He tests a point on the other side of the parabola where the function is decreasing.  
	* **Result:** The calculation yields a negative slope of approximately **\-22**
- Zero slope: The bottom of the parabola (around x ≈ ⅔),  bumping the input in either direction does not change the output significantly, meaning the slope is **0**.

### Derivative of a function with multiple inputs
This is a crucial step towards understanding neural networks, which are essentially **massive mathematical expressions with many inputs**. partial derivatives describing how the output responds to changes in each specific input variable. This connects directly to neural network training, where we need to know how the **“loss” changes with respect to thousands of individual “weights”**.

**The Multi-Variable Function**: We can introduce a function with 3 **scalar inputs** $(a, b, c)$ and a single output $d$. 
- The expression: $d = a \times b + c$ 
- The inputs: $a$ = 2.0, $b$ = -3.0, $c$ = 10.0
- The output: $2.0 \times (-3.0) + 10.0 = 4$
**Partial Derivatives (The “Sensitivity” of each input)**: The goal is to determine the derivative of the output ***d*** with respect to **each** of the inputs (***a***, ***b***, and ***c***) individually. This answers the question: *"If I wiggle just this one input slightly, how does the final output change?".*
A **partial derivative** measures **how the function changes with respect to one variable**, while keeping the other **constant**. - it’s like you’re on a hilly landscape, checking the slope in just one direction ($x$ or $y$) while ignoring the other. Example:
- Function： $f(x,y) = 3x^2y +2xy^2 + y$
- **Step 1**: we differentiate **term by term**, like in normal calculus, but treat $y$ as a **constant** (calculate the derivative of the function with respect to x):  For example:$$f(x,y) = \underbrace{3x^2y}_{\text{has } x} + \underbrace{2xy^2}_{\text{has } x} + \underbrace{y}_{\text{no } x \text{ (constant)}}$$
	Now differentiate: $$\frac{\partial f}{\partial x} = \frac{\partial}{\partial x}(3x^2y) + \frac{\partial}{\partial x}(2xy^2) + \frac{\partial}{\partial x}(y)$$
	$\frac{\partial}{\partial x}(3x^2y) = 6xy$
	$\frac{\partial}{\partial x}(2xy^2) = 2y^2$
	$\frac{\partial}{\partial x}(y) = 0$ ($y$ is treated as constant)
	So:
$$\frac{\partial f}{\partial x} = 6xy + 2y^2$$
We calculates these derivatives numerically (using the "nudge" h) and then verifies them analytically.
**A. Derivative with respect to a ($\frac{\partial d}{\partial a}$)**
	**Intuition:** If we increase $a$ slightly, what happens to $d$?
	Since $a$ is multiplied by $b$ (which is **-3.0**), increasing $a$ makes the product $a \cdot b$ more negative (smaller).
	Therefore, the output $d$ will go down. We expect a negative slope.
	**Numerical Result:** Nudging $a$ yields a slope of **-3.0**.
	**Calculus Check:** The derivative of $(a \cdot b + c)$ with respect to $a$ is just $b$. Since $b = -3.0$, the slope is correct. 
	**Demonstration**: Here, $a$ is our variable $x$, and $b$ and $c$ are treated as constants. We change $a$ to $(a + h)$:
		**1. Set up the formula:**$$\frac{f(a + h, b, c) - f(a, b, c)}{h}$$**2. Plug in the function:**$$\frac{[(a + h) \cdot b + c] - [a \cdot b + c]}{h}$$**3. Expand and simplify:**$$\frac{a \cdot b + h \cdot b + c - a \cdot b - c}{h}$$**4. Cancel out opposing terms:** Notice that $a \cdot b$ and $-a \cdot b$ cancel out, and $+c$ and $-c$ cancel out. $$\frac{h \cdot b}{h} = b$$As $h$ approaches 0, the result is simply **$b$**.
	
**B. Derivative with respect to b ($\frac{\partial d}{\partial b}$)**
	**Intuition:** If we increase $b$ slightly, what happens to $d$?
	Since $b$ is multiplied by $a$ (which is **2.0**), increasing $b$ (making it less negative) adds more to the total.
	Therefore, the output $d$ will go up. We expect a positive slope.
	**Numerical Result:** Nudging $b$ yields a slope of **2.0**.
	**Calculus Check:** The derivative of $(a \cdot b + c)$ with respect to $b$ is just $a$. Since $a = 2.0$, the slope is correct.
	**Demonstration**: Same as derivative with respect to a

**C. Derivative with respect to c ($\frac{\partial d}{\partial c}$)**
* **Intuition:** If we increase $c$ slightly, what happens to $d$?
* The term $c$ is simply added to the rest of the expression. If $c$ goes up by a tiny amount, $d$ goes up by that exact same amount.
* **Numerical Result:** Nudging $c$ yields a slope of **1.0**.
* **Calculus Check:** The derivative of $(a \cdot b + c)$ with respect to $c$ is 1.
* **Demonstration**: Here, $c$ is our variable $x$, and $a$ and $b$ are treated as constants. We change $c$ to $(c + h)$:
		**1. Set up the formula:**$$\frac{f(a, b, (c + h)) - f(a, b, c)}{h}$$**2. Plug in the function:**$$\frac{[a\cdot b + (c+h)] - [a \cdot b + c]}{h}$$**3. Expand and simplify:**$$\frac{a \cdot b + (c+h) - a \cdot b - c}{h}$$**4. Cancel out opposing terms:** Notice that $a \cdot b$ and $-a \cdot b$ cancel out, $$\frac{(c+h) - c}{h} = \frac{h}{h} = 1$$As $h$ approaches 0, the result is simply **1**.

### Creating the Value Object and its visualization
```Python
class Value:

  

  def __init__(self, data, _children=(), _op='', label=''):

    # Stores the actual scalr number (e.g., 2.0)

    self.data = data

    # Stores the derivative of the fianl output with respect to this value.

    # It starts at 0.0

    self.grad = 0.0

    # A function that will calculate the gradient for the inputs of this node.

    # By deafult, it does nothing (for leaf nodes like input data)

    self._backward = lambda: None

  

    # A set of "children" (input nodes) that produced this value.

    # For example, if d = a * b, then d saves the pointers to a and b.

    # This links the nodes together to form a Directed Acyclic Graph (DAG)

    self._prev = set(_children)

    # String variable to track which operation created the value (e.g., '+' or '*'); This helps visualizing the graph later.

    self._op = _op

    self.label = label

  

  # return a printable representation of the object

  def __repr__(self):

    return f"Value(data={self.data})"

  

  def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other)

    # by passing (self,other) tuple, you are recording that "This new value out was created by combing self and other"

    # example: a+b --> a = self, b = other

    # This line of code calculates the actual sum of the data.

    # It creates a new node (out) in the computation graph, remembering that self and other are its parents

    out = Value(self.data + other.data, (self, other), '+')

    # We write the recipe for the derivative

    # When this function runs later, it will take the gradient from the output (out.grad)

    # and copy it to both input gradients (self.grad and other.grad)

    def _backward():

      # x+y --> partial derivative of f/x = 1 and f/y = 1

      # We multiply the local derivative (1.0) by the upstream gradient (out.grad)

      # and accumulate it into the input's gradient (self.grad)

      # Role: Gradient Distributor (or Router).

      # Behavior: It takes the upstream gradient (the signal coming from the loss function)

      # and copies it equally to all of its inputs.

      # since x and y contribute equally to the sum (1-to-1 ratio),

      self.grad += 1.0 * out.grad

      other.grad += 1.0 * out.grad

    # We store the recipe on the output node

    # When you call .backward() on the final loss, the system goes back

    # and executes this stored recipe. This function is saved to be executed later.

    out._backward = _backward

  

    return out

  

  def __mul__(self, other):

    other = other if isinstance(other, Value) else Value(other)

    out = Value(self.data * other.data, (self, other), '*')

  

    def _backward():

      # Role: Gradient Switcher (or Scaler)

      # It swaps the values of the inputs and multiplies them by the upstream gradinet.

      # because the Gradient of x depdens on the value of y and the gradient of y depends on the value of x

      # if y is very large, a tiny change in x will have a huge effect on the output z.

      # Therefore, the gradient for x must be scaled by y.

      self.grad += other.data * out.grad

      other.grad += self.data * out.grad

    out._backward = _backward

  

    return out

  

  def tanh(self):

    x = self.data

    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

    out = Value(t, (self, ), 'tanh')

  

    def _backward():

      self.grad += (1 - t**2) * out.grad

    out._backward = _backward

  

    return out

  

  def backward(self):

    # This list topo ends up ordered from inputs to output.

    # Example: If d = a + b, the list will look like [a, b, d].

    # a and b are added before d.

    topo = []

    visited = set()

    # Recursive function

    def build_topo(v):

      # It marks the current node v as visited so we don't process it twice

      if v not in visited:

        visited.add(v)

        # It looks at all _children (The inputs to this node)

        # and recursively calls build_topo() on them first.

        for child in v._prev:

          build_topo(child)

        # only after all children are processed

        # does it append v to the topo list

        topo.append(v)

    build_topo(self)

    # Sets the gradient of the final output node (the one you called .backward() on) to 1.0

    # The derivative of a variable with respect to itself is always 1,

    # without this initial push, all subsequent multiplication would result in zero. (0*....=0)

    self.grad = 1.0

    # reversed(topo): We flip the list order.

    # reveserd ecause it must flow backwards

    for node in reversed(topo):

      node._backward()
```

### Manual backpropagation Example #1 (Chain Rule)
Instead of relying on the node to calculate derivative automatically, he manually derives and calculates the gradient for every node in the graph to demonstrate exactly how the Chain Rule works.
**The Expression Graph:** 
- **Inputs**: $a = 2.0, b = -3.0, c = 10.0$  
- **Intermediate values:**  
	* $e = a * b = -6.0$  
	* $d = e + c = 4.0$  
	* $f = -2.0$ (new variable he introduced)  
- **Final** **Output**: $L = d * f = -8.0$

**The Goal: Find the Gradients**  
The goal is to find the derivative of the final output ***L*** with respect to every other variable in the graph (weights and data). This tells us: *If we wiggle this variable slightly, how much does L change?*

#### Chain Rule
if a variable $z$ depends on the variable $y$. Which itself depends on the variable $x$ (this is, $y$ and $z$ are dependent variables), then z depends on x as well, via the intermediate variable y. In this case, the chain rule is expressed as: $$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$
Intuitively, the chain rule states that knowing the instantaneous rate of change of $z$ relative to $y$ and that of $y$ relative to $x$ allows one to calculate the instantaneous rate of change of $z$ relative to $x$ as the product of the two rates of change.
As put by George F.Simmons: “ if a car travels twice as fast as a bicycle and the bicycle is four times as fast as a walking man. then the car travels 2 * 4 = 8 times as fast as the man.”

#### Step-by-step backpropagation (walking backwards)
![[Pasted image 20261007010014.png]]

We starts at the end (L) and works backward to the inputs (a,b,c).

- **Step A: Gradient of L with respect itself (dL/dL)**  
	* **Math**: If you change L by a tiny amount h, L changes by exactly h. The slope is 1.  
	* **result**: `L.grad` = 1.0  
- **Step B: Gradient of d and f (Multiplication node)**  
	* **Equation**: ***L \= d \* f***  
	* **Math**: For a multiplication z \= x \* y, the derivative with respect to x is just y →  **Demonstration**: $$\frac{f(x+h)-f(x)}{h} \rightarrow \frac{(x+h) \cdot y - x \cdot y}{h} \rightarrow \frac{xy + hy - xy}{h} \rightarrow \frac{hy}{h} \rightarrow y$$
		- $\frac{dL}{dd} = f = -2.0$
		- $\frac{dL}{df} = d = 4.0$
	* **Intuition**: The derivative of a multiplication operation is just a “swap” of values.  
	* **Result**: `d.grad` = -2.0, `f.grad` = 4.0 
- **Step C: Gradient of c and e (Addition Node)**  
	* **Equation**: $d = c+e$ 
	* **The Chain Rule**: To get ***dL/dc***, we multiply the local derivative ( ***dd/dc***) by upstream gradient (***dL/dd***)  
		- **Chain Rule** is the Mathematical tool that allows us to connect the end (root) of the graph (the Loss L) back to the beginning (Inputs like c), even if they are separated by several steps. The chain rule tells us that to **bridge the gap from L to c**, we simply **multiply the “sensitivities” of each step together**.  
		- **Local Derivative:** if $d = c+e$ , increasing $c$ by h increases $d$ by exactly h. So the slope is 1.0.  $\frac{dd}{dc}$ = 1 → **Demonstration**: $$ \frac{f(x+h)-f(x)}{h} \rightarrow \frac{c+h+e - (c+e)}{h} \rightarrow \frac{c + h + e - c - e}{h} \rightarrow \frac{h}{h} = 1 $$
		- **Calculation**:  $\frac{dL}{dc} =  \frac{dL}{dd} * \frac{dd}{dc}$  → we already know the dL/dd from previous calculation ( \-2.0), and we also calculated our local derivative  $\frac{dd}{dc}$ (1), so the calculation become → $(-2.0) * (1) = -2.0$.  
	* **Intuition**: The addition node is a “gradient distributor”. It simply takes the incoming gradient (-2.0) and routes it equally to all its inputs.  
	* **Result**: c.grad \= \-2.0, e.grad \= \-2.0  
- **Step D: Gradient of $a$ and $b$ (Multiplication Node)**
	* **Equation:** $e = a \cdot b$
	* **The Chain rule:** We need $\frac{dL}{da}$
  $$\frac{dL}{da} = \frac{dL}{de} \text{ (upstream)} \cdot \frac{de}{da} \text{ (local)}$$
  $$\frac{dL}{da} = \frac{dL}{dd} \cdot \frac{dd}{de} \cdot \frac{de}{da} \rightarrow \frac{dL}{de} = \frac{dL}{dd} \cdot \frac{dd}{de}$$
		* **Local derivative:** Since $e = a \cdot b$, the derivative with respect to $a$ is $b$ (which is $-3.0$).
		* **Calculation:** $\frac{dL}{da} = -2.0 \text{ (upstream)} \cdot -3.0 \text{ (local)} = 6.0$
		* **Calculation:** $\frac{dL}{db} = -2.0 \text{ (upstream)} \cdot 2.0 \text{ (local)} = -4.0$
	* **Result:** `a.grad` $= 6.0$, `b.grad` $= -4.0$.

Backpropagation reuses the **stored upstream gradient** **instead of recalculating the entire chain from scratch** is essentially an application of **Dynamic Programming**.   
In the specific context of calculus and computer science, it is formally known as **Reverse-Mode Automatic Differentiation** (or **Reverse Accumulation**).
- **Dynamic Programming** is a method for **solving complex problems** by **breaking them down into simpler subproblems** and storing the results (memoization) so you never have to solve the same subproblem twice.
	* **The Problem:** Calculate $\frac{dL}{da}$
	* **The Naive Way:** Calculate $\frac{dL}{dd} \cdot \frac{dd}{de} \cdot \frac{de}{da}$
	* **The DP Way:**
		  1. Calculate $\frac{dL}{de}$ once.
		  2. Store it in `e.grad`.
		  3. When you need $\frac{dL}{da}$, just look up `e.grad` and multiply by one small local step.
	- Because we "cached" the result at node $e$, we turned a complex chain multiplication into a single step. This turns an exponential problem into a linear one.
- **Reverse-Mode Differentiation (The Math Method)**: In the world of Automatic Differentiation (Autograd), this specific direction of reuse is called **Reverse Accumulation**.
	* **Forward Mode:** You would start at inputs $a, b$ and push derivatives forward toward $L$. This is bad for neural networks because you have millions of inputs but only one output (Loss).
	* **Reverse Mode:** You start at the single output $L$ and propagate backwards. Because the upstream gradient sums up everything that happened later in the graph, you get the gradients for all inputs in one single sweep.

#### Multiplication and Addition nodes are fundamental budling block

**The Addition Node ($+$)**
- **Forward Pass** (Data Flow)
	* **Function:** It sums the inputs together.
	* **Equation:** $z = x + y$
	* **Role:** It combines signals additively.
 - **Backward Pass** (Gradient Flow)
	* **Role:** The "Gradient Distributor" (or Router).
	* **Behavior:** It takes the upstream gradient (the signal coming from the loss function) and copies it equally to all of its inputs.
	* **Intuition:** Since $x$ and $y$ contribute equally to the sum (1-to-1 ratio), they both receive the full force of the gradient. The node acts like a wire splitter.
	* **Math:**$$ \text{Input gradient}_x = 1.0 \cdot \text{Upstream Gradient}$$$$\text{Input gradient}_y = 1.0 \cdot \text{Upstream Gradient}$$
**The Multiplication Node ($*$)**
- **Forward Pass (Data Flow)**
	* **Function:** It multiplies the inputs.
	* **Equation:** $z = x \cdot y$
	* **Role:** It interacts signals multiplicatively (scaling or gating).
 - **Backward Pass (Gradient Flow)**
	* **Role:** The "Gradient Switcher" (or Scaler).
	* **Behavior:** It swaps the values of the inputs and multiplies them by the upstream gradient.
	  * The gradient for $x$ depends on the value of $y$.
	  * The gradient for $y$ depends on the value of $x$.
	* **Intuition:** If $y$ is very large, a tiny change in $x$ will have a huge effect on the output $z$. Therefore, the gradient for $x$ must be scaled by $y$.
	* **Math:**$$\text{Input Gradient}_x = y \cdot \text{Upstream Gradient}$$$$\text{Input Gradient}_y = x \cdot \text{Upstream Gradient}$$
### Single Optimization Step (Manual gradient Ascent)
We want to increase the output $L$. Currently, the output $L$ is $-8.0$. Karpathy wants to change the input values $(a,b,c,f)$ slightly so that the $L$ becomes more positive.
We only update the **leaf nodes** ($a,b,c,f$) because they are **independent variables** (These are the **Weights** and **Biases.** We control them directly). The **intermediate** nodes ($d,e$) are **dependent variables** (These are the “**hidden** **states**” or “**activations**”. We cannot touch them directly; we can only influence them by changing the **weights** that produced them).
- The value of $e$ is defined strictly as $a*b$. It has **no independent existence**. It is just a **calculation result**.  BUT if $a$ is the Input Data of the neural network, then it’s FIXED (FROZEN) that we are not going to change. Because it represents the reality you are trying to learn from. 
- If you manually change $e$ from $-6.0$ to $-5.0$ without changing $a$ or $b$, the mathematical graph becomes **broken**. because $a*b=2.0*-3.0=-6.0 \neq -5.0$. To legitimately change $e$, you must change the things that caused it ($a$ or $b$).

**Strategy: “Nudge” in the direction of the gradient;** note:  it’s **gradient Ascent** here. (different than gradient descent). **Gradient Ascent**: you want to find the maximum value of a function (climb to the top). **Gradient Descent**: you want to find the minimum value of a function (descend to the bottom of the valley).
- **Gradient Ascent**: You **add** the gradient: $$w_{new} = w_{old}+(stepsize \times gradient)$$
- **Gradient Descent**: You add **subtract** gradient: $$w_{new} = w_{old}-(stepsize \times gradient)$$
We can manually updates the **leaf** nodes using a step size of $0.01$:
```python
a.data += 0.01 * a.grad
b.data += 0.01 * b.grad
c.data += 0.01 * c.grad
f.data += 0.01 * f.grad

# Forward pass again
e = a * b
d = e + c
L = d * f

print(L.data) #-7.286496
```
After changing the inputs, the graph is stale (the old output `L` is no longer valid). He effectively runs the **forward pass** again by re-calculating the intermediate values. Results the new $L$ is roughly $-7.2$ (it went up from -8.0).

#### Manual Backpropagation example #2: a neuron
While biological neurons are complex, deep learning uses a simplified mathematical model:
- **Synapses (Weights):** Inputs (***x***) are multiplied by weights (***w***). This represents the "synaptic strength."  
- **Cell Body (Sum):** The weighted inputs are summed up, and a **bias** (***b***) is added. This bias controls the "trigger happiness" of the neuron.  
- **Activation Function:** The total sum is passed through a squashing function (non-linearity) to produce the final output. We chooses **tanh** (hyperbolic tangent), which squashes values between -1 and 1.
The formula for the neuron's output is: $$\text{Output} = \tanh \left(\sum(w_i \cdot x_i) + b\right)$$tanh’s formula in terms of the exponential function (e): $$\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$derivative of tanh: **1- _tanh(x)^2_** .
Setting up the example
- **Inputs: $x_1$ = $2.0$, $x_2$ = $0.0$
- **Weights: $w_1$ = $-3.0$, $w_2$ = $1.0$** 
- **Bias: $b$ = $6.88137$** (chosen specifically to make the numbers work out nicely later).
- **The Expression:** 
	1. x1w1 = $x_1 * w_1$ (Result: -6.0).
	2. x2w2 = $x_2 * w_2$ (Result: 0.0)
	3. x1w1x2w2 = $x_1*w1 + x_2*w_2$ (Result: -6.0) 
	4. $n$ = x1w1x2w2 + b (Result: 0.88137...) 
	5. $o$ = $tanh(n)$ (Final Output: 0.7071).
Since `micrograd` didn't have a `tanh` operation yet, he adds it to the `Value` class.
- Instead of building it from atomic pieces (exponentiation, division) right away, he implements it as a single "black box" operation.
- This demonstrates that you can define the level of abstraction for your operations anywhere you like, as long as you know the **local derivative**.
	- Instead of creating 10 small Python objects and tracking their history, you create 1 object. This **saves memory and speeds up calculation**. Sometimes, breaking things down into tiny steps causes **math errors** (like dividing by a very small number). If you use the abstract "black box" version, you can often use a mathematically simplified formula that avoids these errors.

### How the Signal flow?
Backpropagation is essentially a long chain of multiplication. For example the Local derivative of tanh function is always <= 1. Because the gradient of tanh is calculated like this: $\frac{dy}{dx} = 1-y^2$, since $tanh(y)$ always outputs a value between -1 and 1, let's look at what happens to the derivative ($1-y^2$): 
- Maximum value: if y \= 0, the derivative is 1 \- 0 \= 1\.  
* **Typical Value:** If ***y*** is anything else (e.g., **0.5**), the derivative is less than 1 (e.g., ***1 \- 0.5 \= 0.5***).  
* **Minimum Value:** If y is saturated at 1 or \-1, the derivative is 1 \- 1 \= 0\.
During backpropagation, the gradient “flowing” through the network is just a chain of multiplications. When the gradient arrives at a `tanh` gate from a later layer, it is **multiplied** by the local derivative of that `tanh` gate. $gradient_{out} = gradient_{in} * LocalDerivative$. Because the Local derivative is always between 0 and 1:
- You are always multiplying the incoming signal by a number like ***0.9,*** ***0.5***, or ***0.01***.  
- Mathematically, multiplying a number by a factor \<= 1 will always result in a number with an equal or smaller magnitude.
You are mathematically guaranteed to **lose signal magnitude** (or at best maintain it, but never increase it) as you move backward. If you imagine a network with 5 layers, the gradient at the first layer looks something like this: $$Grad_1 = Grad_{last} \times (\le1)\times (\le1)\times (\le1)\times (\le1)$$If those numbers are even slightly less than 1 (E.g., 0.9), the result shrinks exponentially. If they are small, the gradient effectively becomes zero very quickly. The result:
- **Last Layer (Output):** Receives the raw error signal. This is the "loudest" or "most intense" the signal will ever be.  
- **Middle Layers:** The signal has been dampened by several multiplications of numbers \<1.  
- **First Layer (Input):** Receives the "whisper" of the original signal.
The signal **starts strong at the end** and **keeps reducing** as it flows to the front. The deeper the network, the worse this reduction becomes.

#### Manual backpropagation (The "backward Pass")
We starts at the end (the output `o`) and calculates the gradient for every node backward to the inputs.

- **Output Gradient (`o.grad`):**  
	* The base case is always **1.0**.  
- The Tanh Gradient (`n.grad`)  
	* **Math:** The derivative of **tanh(x)** is $1 - tanh(x)^2$.  We calculate the local derivative  $\frac{do}{dn}= 1 - tanh(n)^2$
	* **Calculation:** Since the output ***o = tanh(n)***, the local derivative is ***1 - o^2***  
	* With ***o ≈ 0.7071, o^2 \= 0.5***.  
	* The gradient is ***1 \- 0.5 \= 0.5***.  
	* *Chain Rule:* ***1.0 (upstream) \* 0.5 (local) \= 0.5***.  
	* ![[Pasted image 20261007170742.png]]
- The Addition Gradients (`+`)   
	* The gradient of **0.5** flows into the `+` node.  
	* As established previously, the + node is a "distributor." It copies the gradient to both inputs (b and sum of weights).  
	* Result: `b.grad` \= 0.5, `x1w1x2w2.grad` \= 0.5.  
	* ![[Pasted image 20261007170753.png]]![[Pasted image 20261007170816.png]]
- The Multiplication Gradients (`*`)  
	* He calculates gradients for the weights ($w_1, w_2$) and inputs ($x_1, x_2$).  
	* **For w\_2:** The upstream gradient is 0.5. The local derivative is $x_2$ (which is 0.0).  
		- **Result**: $0.5 * 0.0 = 0.0$.  
		- *Intuition:* Since the input $x_2$ was $0$, changing the weight $w_2$ has **no effect** on the output. Therefore, the gradient is zero.  
	* **For w\_1:** The upstream gradient is 0.5. The local derivative is $x_1$ (which is 2.0).  
		- **Result:** $0.5 * 2.0$ = 1.0.

### Implementing the backpropagation in python
To automating the process by embedding the logic directly into the **Value** class. Instead of calculating derivatives *after* the graph is built, we modifies the operations (`+`, `*`, `tanh`) to **store** the recipe for calculating their own gradients *while* the graph is being built.
We add `_backward` function to every `operation function` in Value object
```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)

    # by passing (self,other) tuple, you are recording that "This new value out was created by combing self and other"

    # example: a+b --> a = self, b = other

    # This line of code calculates the actual sum of the data.

    # It creates a new node (out) in the computation graph, remembering that self and other are its parents

    out = Value(self.data + other.data, (self, other), '+')

    # We write the recipe for the derivative

    # When this function runs later, it will take the gradient from the output (out.grad)

    # and copy it to both input gradients (self.grad and other.grad)

    def _backward():

      # x+y --> partial derivative of f/x = 1 and f/y = 1

      # We multiply the local derivative (1.0) by the upstream gradient (out.grad)

      # and accumulate it into the input's gradient (self.grad)

      # Role: Gradient Distributor (or Router).

      # Behavior: It takes the upstream gradient (the signal coming from the loss function)

      # and copies it equally to all of its inputs.

      # since x and y contribute equally to the sum (1-to-1 ratio),

      self.grad += 1.0 * out.grad

      other.grad += 1.0 * out.grad

    # We store the recipe on the output node

    # When you call .backward() on the final loss, the system goes back

    # and executes this stored recipe. This function is saved to be executed later.

    out._backward = _backward

  

    return out
```
For `example_F1`: 
```python
# inputs x1,x2

x1 = Value(2.0, label='x1')

x2 = Value(0.0, label='x2')

# weights w1,w2

w1 = Value(-3.0, label='w1')

w2 = Value(1.0, label='w2')

# bias of the neuron

b = Value(6.8813735870195432, label='b')

# x1*w1 + x2*w2 + b

x1w1 = x1*w1; x1w1.label = 'x1*w1'

x2w2 = x2*w2; x2w2.label = 'x2*w2'

x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

n = x1w1x2w2 + b; n.label = 'n'

o = n.tanh(); o.label = 'o'
```

```python
  def tanh(self):

    x = self.data

    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

    out = Value(t, (self, ), 'tanh')



    def _backward():

      self.grad += (1 - t**2) * out.grad

    out._backward = _backward


    return out
```
`o = n.tanh()` →  in the tanh() function,  ***n*** will be passed as self (`n.tanh()`), so variable ***x*** will save ***n’s*** data. ***t*** will be intermediate value that will execute tanh function  and we will create a new Value object out for it to be saved as ***o*** later, the children of the ***o*** is **n** here. we save how should we calculate the gradient of the children (***n***) is defined in the `_backward()` function, and we save it in the attribute of the Value object **o** to be ready to be retrieved later when we call `o.backward()`.
- **`o._backward` is not for $o$. It is for $n$ (parent).**
- backward() method starts by setting o.grad = 1.0.  Because o is the end of the road. The derivative of a variable with respect to itself is 1\.

Why saving the “**Function object**” works? In Python, a function object (a closure) carries a **backpack containing the variables** that were present when it was defined. When `o = n.tanh()` ran earlier:
1. A function `_backward` was created.  
2. inside that function’s “backpack” (Closure), it permanently saved references to `n` (as `self`) and `o` (as `out`).  
3. We assigned `o._backward` to hold this specific function object
When `o.backward()` runs later:
4. It picks up that function object  
5. It puts parentheses () after it to tell python “Run this code now.”  
6. The function opens its backpack, finds **n** and **o**, and performs the update.

#### Automating Backpropagation with Topological Sort
Previously we calling `node._backward()` for every single node in reverse order is not feasible.  Andrej introduces a method to automate this process while ensuring the correct order of operations.
Because of Chain Rule, we cannot calculate the gradient of a node until the gradients of everything that consumes it have been calculated. - If `b = a + c`, you must know `b.grad` before you can calculate `a.grad`. If you process nodes in the **wrong order**, the **chain rule breaks** because the upstream gradient will be zero or incomplete.
We can use a graph algorithm called **Topological Sort** to flatten the graph into a list where very node appears only after all its children (dependencies) have been processed. A recursive function `build_topo` that visits children first before adding the parent to the list is implemented. The key behavior lies in the recursion structure:
```python
  def backward(self):

    # This list topo ends up ordered from inputs to output.

    # Example: If d = a + b, the list will look like [a, b, d].

    # a and b are added before d.

    topo = []

    visited = set()

    # Recursive function

    def build_topo(v):

      # It marks the current node v as visited so we don't process it twice

      if v not in visited:

        visited.add(v)

        # It looks at all _children (The inputs to this node)

        # and recursively calls build_topo() on them first.

        for child in v._prev:

          build_topo(child)

        # only after all children are processed

        # does it append v to the topo list

        topo.append(v)

    build_topo(self)

    # Sets the gradient of the final output node (the one you called .backward() on) to 1.0

    # The derivative of a variable with respect to itself is always 1,

    # without this initial push, all subsequent multiplication would result in zero. (0*....=0)

    self.grad = 1.0

    # reversed(topo): We flip the list order.

    # reveserd ecause it must flow backwards

    for node in reversed(topo):

      node._backward()
```
This function effectively says: *"I cannot add myself to the list until I have successfully added all of my dependencies (children) to the list first."*
Immagine a single neuron is build like this `example_F1`, when you call `build_topo(o)`, the computer builds a call stack that dives all the way down to the input leaves before adding anything to the list. The execution starts at the output `o`, goes to `n`, and then hits a **fork in the road**. Node *n* has 2 parents (children in the graph structure)
1. The bias (`b`)
2. The sum of inputs ($x_1*w_1 + x_2*w_2$)
![[Pasted image 20261007220003.png|288]]
The exact sequence of the “Call Stack” (what the computer is thinking) vs. the “Topo List” (what gets written down).
- **Phase 1: The deep Dive (left branch)**  
		1. **Call build\_topo(o):** it calls build\_topo(n).  
		2. **Call build\_topo(n):** it iterates through children. It picks the “Sum” branch first.  
		3. **Dive Deeper**: It keeps calling `build_topo` recursively down through `x1*w1 + x2*w2` \-\> `x2*w2` \-\> `x2` and `w2`.  
		4. **Hit Bottom (Leaves)**: It reaches `w2` and `x2`. They have no children.  
			* **Action:** `w2` is appended. (Index 0 in your list)  
			* **Action:** `x2` is appended. (Index 1\)  
			* **Action:** The parent `x2*w2` is now done. It is appended. (Index 2\)  
- **Phase 2: The middle (Processing the rest of the sum)**  
	1. **Back up slightly**: The recursion moves to the `x1*w1` side.  
	2. **Dive Deeper**: It goes down to `w1` and `x1`.  
		* **Action:** `w1` is appended. (Index 3\)  
		* **Action:** `x1` is appended. (Index 4\)  
		* **Action:** The parent `x1*w1` is appended. (Index 5\)  
	3. **Finish the Sum**: Now both parts of the sum are done.  
		* **Action:** The node `x1*w1 + x2*w2` is appended. (Index 6\)  
- **Phase 3: The "Late" Arrival (Right Branch)**  
	1. **Return to `n`**: The code finally returns to the loop inside `build_topo(n)`. It asks: "Are there any other children?"  
		1. **Found `b`**: Yes, `b` is still waiting\!  
		2. **Call `build_topo(b)`**:  
			- `b` has no children.  
			- **Action:** `b` is appended. (Index 7\)  
- **Phase 4: The Finish Line**  
	1. **Finish `n`**: All children of `n` (the Sum and `b`) are now in the list.  
		* **Action:** `n` is appended. (Index 8\)  
	2. **Finish `o`**: All children of `o` are now in the list.  
		* **Action:** `o` is appended. (Index 9\)
The backward() function is implemented into the **Value** class to automate the backpropagation process.

#### Multivariate Chain Rule
The Multivariate Chain rule is the mathematical foundation for backpropagation when a  **single variable branches out** to  **affect multiple other variables in a computational graph**.  
In standard “single-variable” calculus, if $z$ depends on $y$ and $y$ depends on $x$, the rule is simply: $\frac{dz}{dx} = \frac{dz}{dy} \times \frac{dy}{dx}$.
However, in **multivariate** calculus, if a variable ***x*** influences multiple intermediate variables ($y_1, y_2, …, y_n$), and all of those ***y*** values eventually contribute to the final output ***z,*** the total change in ***z*** with respect to ***x*** is the **sum of the changes through every possible path**. $$ \frac{\partial z}{\partial x} = \sum_{i} \left( \frac{\partial z}{\partial y_i} \times \frac{\partial y_i}{\partial x} \right) $$Example: Imagine you have the following relationships:
1. $z = y_1 + y_2$
2. $y_1 = x^2$
3. $y_2 = 3x$
to find the $\frac{dz}{dx}$ we calculate the contribution from each path and add them together:
- Path 1 (via $y_1$): $\frac{\partial z}{\partial y_1} \cdot \frac{\partial y_1}{\partial x} = 1 \cdot (2x) = 2x$
- Path 2 (via $y_2$): $\frac{\partial z}{\partial y_2} \cdot \frac{\partial y_2}{\partial x} = 1 \cdot 3 = 3$
Total Derivative: $\frac{dz}{dx}$ = $2x+3$. A real world example of this rule because it appears in two different lines of the forward pass:
```python
# exponential all the logits, shape[32,27]
counts = norm_logits.exp()
# calculate the denominator and its inverse, shape [32,1]
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact..., [32,1]
probs = counts * counts_sum_inv
logprobs = probs.log()
# selects the log-probability of the correct character (YB)
# n = number of batch
# if the model is performing perfectly, the probability for the correct class
# would be 1.0, the log(1.0) would be 0, and the loss would be 0.
loss = -logprobs[range(n), Yb].mean()
```
In the code, counts is used in two places:
- Path 1 (numerator) `probs = counts * counts_sum_inv` (used as the numerator in the softmax, it doesn’t mean this entire expression is numerator, this entire expression itself computes the entire probability).  
- Path 2 (Denominator): `counts_sum = counts.sum(1, keepdims=True)` (used to calculate the sum for the denominator).

**Deriving Path 1**: For the line `probs = counts * counts_sum_inv`, we take the partial derivative with respect to `counts`:
* The derivative of `a*x` with respect to `x` is `a`.  
* Therefore, the local gradients is `counts_sum_inv`.  
* Applying the chain rule: `dcounts_path1 = counts_sum_inv * dprobs`.

**Deriving Path 2**: For the line `counts_sum = counts.sum(1, keepdims=True)`, we follow the gradient through the summation:
* In the forward pass, `counts_sum` is just the sum of `counts` across the row.  
* The derivative of a sum ($x_1 + x_2 + …$) with respect to any $x_i$ is $1$.  
* Therefore, the gradient `dcounts_sum` just needs to be "broadcasted" back to all the elements that were summed.  
* Applying the chain rule: `dcounts_path2 = torch.ones_like(counts) * dcounts_sum`.
Putting it all together:
```python
dcounts = counts_sum_inv * dprobs  # Path 1  
dcounts += torch.ones_like(counts) * dcounts_sum  # Path 2 (Multivariate Chain Rule)**
```

By doing this, you account for the fact that changing a `count` value affects the loss twice: once by changing the numerator of that specific character's probability, and once by changing the total sum in the denominator for all characters in that row.

#### Breaking up a tanh
In this section, we significantly expands the capabilities of the Value class by moving away from the “black box” implementation of tanh and instead breaks it down into its constituent mathematical operations.
Previously, tanh was a single node with a  hardcoded derivative $(1 - tanh^2)$. We now wants to implement it using its raw definition: $$tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$$To do this, the engine needs to support **Exponentiation** $e^x$ , **division**, and **subtraction**, which is currently lacks.

**Implementing Exponentiation(exp)**
He implements $e^x$ as a new operation.
* **Forward:** Uses `math.exp(self.data)`.  
* **Backward:** The derivative of $e^x$ is simply $e^x$.  
  * **Code:** `self.grad += out.data * out.grad`  
  * *Note:* Since out.data *is* **e^x**, he reuses it nicely for the derivative
```python
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    
    return out
```

**Implementing Division and Power**  
Instead of implementing division (`__truediv__`) directly with its own quotient rule derivative, he implements it as a special case of **Power**: $a/b → a * \frac{1}{b} → a * b^{-1}$.
This requires implementing the power function $x^k$ first.
- **Power (__pow__):** He implements support for raising a value to a constant power (integer or float).  
	* **Forward:** self.data \*\* other  
	* **Backward:** He uses the Power Rule from calculus: $nx^{n-1}$.  
	* **Code:** 
```python
  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(self.data**other, (self,), f'**{other}')
    
    def _backward():
        self.grad += other * (self.data ** (other - 1)) * out.grad

    out._backward = _backward
  
    return out
```
- **Division:** He defines \_\_truediv\_\_(self, other) as simply self \* other\*\*-1.
```python
  def __truediv__(self, other): # self / other
    return self * other**-1
```

**Implementing Subtraction**  
Similarly, he implements subtraction as a composite operation.
* **Negation (\_\_neg\_\_):** He implements unary negation (-a) as self \* \-1.  
* **Subtraction (\_\_sub\_\_):** He defines a \- b as a \+ (-b).

With all pieces in place, he rewrites the forward pass for the neuron without using `.tanh()`:
```python
# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# The new explicit implementation of tanh  
e = (2*n).exp()  
o = (e - 1) / (e + 1)
```
- **Visualization:** When he draws the graph now, it is much larger and more complex. Instead of a single `tanh` node, there is a web of adds, multiplies, powers, and exponentiations.  (check `micrograd_lecture_second_half_rougly.ipynb` for the plot)
- **Verification:** He runs backpropagation on this new graph. The gradients for the weights ($w_1, w_2$) comes out identically to the previous run.
By breaking up a tanh, we illustrate the point that the **level at which you implement your operation is totally up to you**. You can implement backward passes for tiny expressions like a single individual **+** or a single $\times$ or you can implement them for $tanh$ which is kind of a composite operation because it's made up of all these more atomic operations. But really all of this is kind of like a fake concept. All that matters is we have some kind of inputs and some kind of an output and this output is a function of the inputs in some way and **as long as you can do forward pass and the backward pass of that little operation it doesn’t matter what that operation is and how composite it is.** If you can write the local gradients you can chain the gradient and you can continue back propagation. So design of what those functions are is completely up to you.

### Building a neural net Library (MLP) in micrograd
We build a modular neural network library on top of the `Value` engine, mimicking the structure of torch.nn. He constructs it hierarchically: **Neuron --> Layer --> MLP**.

Neuron: It takes the number of inputs (nin) and initializes a list of random weights `w` and a single random bias `b` using values between $-1$ and $1$. 
- `__call__`: it computes the dot product of weights and inputs plus the bias: $\sum (w_i \cdot x_i) + b$
```PYTHON
# A Neuron perform a linear transformation followed by
# a non-linearity (Activation function)
# output = tanh(Summation((w_i*x_i)+b))
class Neuron:
  def __init__(self, nin):

    # nin = number of inputs

    # Create a random weight (Value object) for every input

    # Initialized as Value objects. This is crucial.

    # Because it means the computation graph will track operations on them,

    # allowing us to calculate self.w[i].grad later

    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]

    # Create a single bias (Value object)
    self.b = Value(random.uniform(-1,1))

  # __call__ method enables Python programmers ot write classes
  # where the instances behave like functions
  # and can be called like a function --> Example: e = Gfg() , e()
  def __call__(self, x):
    # w * x + b
    # x = input vector
    # 1. pair up weights and inputs (zip), zip used to combine multiple iterable objects
    # into a single iterable of tuples. Each tuple contains elements from the corresponding position of the input iterables
    # 2. Multiply them (wi*xi)
    # 3. Sum them all up, starting with the bias "b" as the initial accumulator, because:
    # sum() has a optional parameter which is the start and by default, start is 0
    # So these elements of this sum will be added on top of zero to begin with
    # but actually we can just start with self.b

    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

    # Apply the activation function to introduce non-linearlity
    out = act.tanh()
    return out

  

  def parameters(self):

    # Return a flat list of all trainable parameters (weights + bias) in this neuron
    # List concatenation, [self.b] to turn it into a list containing one item
    return self.w + [self.b]
```


The `Layer` class is simply a list of independent Neurons. It takes the number of inputs (`nin`) and the number of desired neurons in the layer `nout`. It creates a list of `Neuron` object.
```Python
# A layer is simply a list of Neurons that operate independently on the same input

class Layer:

  def __init__(self, nin, nout):
    # nin = number of inputs coming into this layer
    # nout = number of neurons in this layer (the size of the layer)
    self.neurons = [Neuron(nin) for _ in range(nout)]

  

  def __call__(self, x):
    # pass the input "x" to every neuron in this layer
    # Here is the __Call__ function working
    outs = [n(x) for n in self.neurons]

    # Convenience: if there is only 1 neuron, return the neuron directly
    # Otherwise return the list of neurons
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    # Gather parameters from all neurons in this layer into one flat list.
    # logic: [ [w,b] , [w,b] ] --> [w, b, w, b]
    return [p for neuron in self.neurons for p in neuron.parameters()]
```

MLP Class represents the entire neural network. It is a sequence of Layers where the **output of one layer becomes the input to the next**.
- It takes `nin` (input dimension) and `nouts` (a *list* of integers defining the sizes of all subsequent layers).  It creates a full list of sizes including the input: `sz = [nin] + nouts`.  It iterates through these sizes to create **Layer** objects. Crucially, it matches dimensions.

```python
class MLP:
  
  def __init__(self, nin, nouts):
    # nin = integer, number of inputs into the network
    # nouts = list of integers, defining the size of all subsequent layers
    # Combine input size with output sizes to get the full list of layers sizes
    # Example: nin = 3, nouts = [4,4,1] -> sz = [3,4,4,1]
    # 3 inputs into  2 layers of 4 and produces 1 output

    sz = [nin] + nouts
    
    # Create layer objects iterating through the sizes
    # Layer 1 connects sz[0] to sz[1], Layer 2 connects sz[1] to sz[2], etc.

    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  

  # Mathematical expression grows significantly here, creating a massive DAG of Value Objects

  def __call__(self, x):
    # Forward Pass:
    # Sequentially pass the data through each layer.
    # The input 'x' is updated to be the output of the previous layer.
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    # Gather parameters from all layers into one massive flat list.
    # This is passed to the optimizer to update all weights in the network at once.
    return [p for layer in self.layers for p in layer.parameters()]
```
Example:
```python
x = [2.0, 3.0, -1.0]
# three inputs into two layers of 4, and one output
n = MLP(3, [4, 4, 1])
n(x) # Value(data=0.20450394250203427)
```
**The Loss function (MSE) & Visualization**
Once the network architecture is built, we demonstrate how to measure its performance. We define a tiny "toy" dataset with 4 examples (`xs`) and 4 desired targets (`ys`). The network should output $1.0$ for some inputs and -1.0 for others.![[Pasted image 20261008012152.png]]
We performs a forward pass for all 4 example to get 4 prediction `yred`. We calculates the loss using MSE. Lower the loss = better predictions.
**The "Rabbit Hole" Visualization**:
* When he calls `draw_dot(loss)`, it generates a massive computational graph.  
* This graph connects the final single number (the loss) backwards through:  
	  * The MSE calculation...  
	  * To the 4 forward passes...  
	  * Through all the layers and neurons...  
	  * All the way back to the 41 individual weights and biases (parameters).  
* This visualizes exactly how every single weight in the network mathematically contributes to the final error rate.

#### Training Loop:
```python
for k in range(20):
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

  # backward pass
  # n.parameters() will return a single, flat list containing value objects
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  print(k, loss.data)
```
* **Step 1: Forward Pass**  
  * Calculate predictions (`ypred`) and the `loss` for the current state of the network.  
* **Step 2: Zero Gradients (The "Bug")**  
  * **The Issue:** Karpathy initially forgets this step and encounters a bug where the network trains unpredictably.  
  * **The Reason:** In `micrograd` (and PyTorch), gradients **accumulate** (`+=`). If you don't reset them, the gradients from the current step add onto the gradients from the previous step, resulting in incorrect, massive updates.  
  * **The Fix:** He explicitly sets `p.grad = 0.0` for every parameter before backpropagation.  
* **Step 3: Backward Pass**  
  * Calls `loss.backward()`. This propagates gradients all the way from the final loss number back to every single weight in the network.  
* **Step 4: Update (Gradient Descent)**  
  * He updates every parameter in the list: `p.data += -learning_rate * p.grad`  
  * **Negative Sign:** We subtract the gradient because we want to **minimize** the loss (descend the hill), not maximize it.  
  * **Learning Rate:** He tunes this number (e.g., changing it from `0.1` to `0.05`) to ensure the training is stable—not too slow, but not so fast that it overshoots and explodes.