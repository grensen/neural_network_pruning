# Neural Network Pruning (Unstructured + Structured = Compression) Using C#

## Network Structures
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/network_structures.png?raw=true">
</p>

Neural network pruning is the process of making deep learning models more efficient. There are two main types of neural network pruning: unstructured pruning, where specific "connections or weights" are removed based on the magnitude of the trained weight or other conditions, and structured pruning, which removes "neurons or nodes" along with their connected weights based on specific conditions. The challenge is to build a simplified model while maintaining high accuracy. 

## The Pruning Idea
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/pruning_idea.png?raw=true">
</p>

The downside of pruning is the requirement to store positions for the still working weights. Additionally, this leads to an increase in calculation time. However, despite using only 10-20% of its core size, a pruned network can perform remarkably close to the original. 

## Jagged Array
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/array_type_fixed.png?raw=true">
</p>

To implement a pruning structure in the neural network, we utilize a jagged array. Here, the rows represent each input neuron, and the columns represent the weights connected to the output neuron. We need two jagged arrays, both of the same size: one for the float values of each weight and another int array for the positions to which the output neurons are connected.

## The Demo
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/pruning_demo.png?raw=true">
</p>

## Reference Fully Connected Neural Network
<p align="center">
  <img src="https://github.com/grensen/neural_network_benchmark/raw/main/benchmark.png?raw=true">
</p>

## High Level Code
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/high_level_demo.png?raw=true">
</p>

## High Level Functions
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/high_level_functions.png?raw=true">
</p>

## Saved Network
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/network_file.png?raw=true">
</p>

## Saved Positions
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/positions_file.png?raw=true">
</p>

## Saved Weights
<p align="center">
  <img src="https://github.com/grensen/neural_network_pruning/blob/main/figures/weights_file.png?raw=true">
</p>

[Demo Code](https://github.com/grensen/ML_demos/blob/main/code/neural_network_benchmark.cs),
[Neural Network Benchmark](https://github.com/grensen/neural_network_benchmark)

