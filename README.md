# Neural Network Implementation

A simple feedforward neural network implementation in JavaScript with backpropagation training algorithm.

## Features

- **Feedforward Architecture**: Single hidden layer neural network
- **Backpropagation Training**: Gradient descent optimization with sigmoid activation
- **Matrix Operations**: Custom Matrix class with essential linear algebra operations
- **Bias Support**: Includes bias terms for both hidden and output layers
- **Error Logging**: Optional training progress monitoring

## Architecture

The neural network consists of:
- Input layer (configurable size)
- Single hidden layer (configurable size)
- Output layer (configurable size)
- Sigmoid activation function for all neurons
- Bias terms for hidden and output layers

## Quick Start

### Creating a Neural Network

```javascript
// Create a network with 2 inputs, 4 hidden neurons, and 1 output
const nn = new NeuralNetwork(2, 4, 1);
```

### Training the Network

```javascript
// Train with input array and target output array
nn.train([0.5, 0.8], [0.9]);

// Training loop example
for (let i = 0; i < 10000; i++) {
    nn.train([0, 0], [0]);  // AND gate example
    nn.train([0, 1], [0]);
    nn.train([1, 0], [0]);
    nn.train([1, 1], [1]);
}
```

### Making Predictions

```javascript
// Get prediction for input
const output = nn.feedForward([1, 1]);
console.log(output.data[0][0]); // Access the prediction value
```

## API Reference

### NeuralNetwork Class

#### Constructor
```javascript
new NeuralNetwork(numInputs, numHidden, numOutputs)
```
- `numInputs`: Number of input neurons
- `numHidden`: Number of hidden layer neurons  
- `numOutputs`: Number of output neurons

#### Methods

**feedForward(inputArray)**
- Performs forward propagation through the network
- Returns Matrix object containing output predictions
- `inputArray`: Array of input values

**train(inputArray, targetArray)**
- Trains the network using backpropagation
- `inputArray`: Array of input values
- `targetArray`: Array of target output values

#### Properties
- `inputs`: Current input values (Matrix)
- `hidden`: Hidden layer values (Matrix)
- `weights0`: Input-to-hidden weights (Matrix)
- `weights1`: Hidden-to-output weights (Matrix)
- `bias0`: Hidden layer bias (Matrix)
- `bias1`: Output layer bias (Matrix)

### Matrix Class

#### Constructor
```javascript
new Matrix(rows, cols, data)
```

#### Static Methods
- `Matrix.add(m0, m1)`: Element-wise addition
- `Matrix.subtract(m0, m1)`: Element-wise subtraction
- `Matrix.multiply(m0, m1)`: Element-wise multiplication
- `Matrix.dot(m0, m1)`: Matrix multiplication (dot product)
- `Matrix.transpose(m0)`: Matrix transpose
- `Matrix.map(m0, function)`: Apply function to each element
- `Matrix.convertFromArray(array)`: Convert 1D array to Matrix

#### Instance Methods
- `randomWeights()`: Initialize with random values between -1 and 1

## Configuration

### Error Logging
```javascript
const LOG_ON = true;      // Enable/disable error logging
const LOG_FREQ = 20000;   // Log error every N iterations
```

## Example: XOR Problem

```javascript
// Create network
const nn = new NeuralNetwork(2, 4, 1);

// Training data for XOR
const trainingData = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] }
];

// Train the network
for (let epoch = 0; epoch < 50000; epoch++) {
    for (let data of trainingData) {
        nn.train(data.input, data.target);
    }
}

// Test the network
console.log("0 XOR 0:", nn.feedForward([0, 0]).data[0][0]);
console.log("0 XOR 1:", nn.feedForward([0, 1]).data[0][0]);
console.log("1 XOR 0:", nn.feedForward([1, 0]).data[0][0]);
console.log("1 XOR 1:", nn.feedForward([1, 1]).data[0][0]);
```

## Technical Details

### Activation Function
The network uses the sigmoid activation function:
```
σ(x) = 1 / (1 + e^(-x))
```

### Learning Algorithm
- **Forward Pass**: Input → Hidden → Output
- **Backward Pass**: Calculate errors and update weights using gradient descent
- **Weight Updates**: Weights adjusted proportional to error gradients
- **Bias Updates**: Bias terms updated alongside weights

### Weight Initialization
Weights are randomly initialized between -1 and 1 using `Math.random() * 2 - 1`.

## Limitations

- Single hidden layer architecture
- Fixed sigmoid activation (no ReLU, tanh options)
- No learning rate parameter (implicitly 1.0)
- No momentum or advanced optimization
- No regularization techniques
- Basic error logging only

## Browser Compatibility

This implementation uses ES6 classes and should work in modern browsers. For older browser support, transpilation with Babel may be required.

