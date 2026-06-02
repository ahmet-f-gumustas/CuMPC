# CuMPC

**CuMPC** is a CUDA-accelerated **Model Predictive Control (MPC)** implementation for **ROS 2**.

The goal of the project is to move the heavy numerical work of MPC — building the
prediction horizon, evaluating the cost function, and solving the underlying
optimization problem — onto the GPU, so the controller can run at high rates on
systems with many states, long horizons, or tight real-time deadlines where a
CPU-only solver would struggle.

## Overview

A standard MPC loop solves a constrained optimization problem at every control
step: predict the system's behaviour over a finite horizon, minimize a cost
function, and apply the first control input. The compute cost grows quickly with
the horizon length and state dimension. CuMPC offloads the parallelizable parts
of this loop to CUDA kernels and exposes the controller as a ROS 2 node, so it
plugs into an existing robotics stack.

## Features

- CUDA kernels for the parallelizable parts of the MPC loop (prediction, cost evaluation, solver iterations)
- ROS 2 node wrapping the controller (subscribes to state, publishes control commands)
- Configurable horizon, model, cost weights, and constraints

## Requirements

- CUDA Toolkit (with a CUDA-capable GPU)
- ROS 2
- C++17 (or newer) compiler
- CMake

## Status

Work in progress. APIs, node names, and parameters are still evolving.

## License

Licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
