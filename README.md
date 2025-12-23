# Fuzzy-Based Intelligent Control System for Vacuum Suction Optimization
An explainable intelligent system to optimize energy usage based on surface and dust conditions.

Most vacuum cleaners operate with fixed or manually adjusted suction power, 
leading to unnecessary energy consumption on light surfaces and insufficient cleaning on heavier conditions.

This project implements an **intelligent control system using fuzzy logic** 
to dynamically adjust vacuum suction power based on **surface characteristics** and **dust levels**, 
providing a balance between cleaning effectiveness and energy efficiency.

## Problem Statement
Manual or static suction settings fail to adapt to varying floor surfaces and dust conditions, 
resulting in either wasted energy or suboptimal cleaning performance.

## Approach
This system applies **fuzzy logic inference** to handle uncertainty in real-world conditions.
Two input variables are evaluated:
- Floor surface condition (smooth to rough)
- Dust level (low to high)

The output is a dynamically adjusted suction power level that responds intelligently 
to environmental changes without rigid thresholds.

## System Design
The system is designed using a **Mamdani-type fuzzy inference system** consisting of:
- Fuzzification of input variables
- Rule-based inference engine
- Defuzzification using a weighted average method

The design emphasizes **interpretability**, allowing each decision to be traced back 
to specific rules and membership functions.

## Validation
The system output has been validated by comparing:
- Manual fuzzy logic calculations
- Programmatic results from the Python implementation

For identical input values, both methods produce **consistent output results**, 
confirming the correctness of the inference and defuzzification process.

## Implementation
The system is implemented in Python using:
- `NumPy` for numerical operations
- `scikit-fuzzy` for fuzzy logic processing
- `Matplotlib` for membership function visualization

The main program accepts user input for surface and dust conditions, 
then computes the suction power level through fuzzification, inference, and defuzzification.

## How to Run
Follow these steps to run the system locally:

1. Clone the repository
   ```bash
   git clone https://github.com/azizi009/fuzzy-suction-optimization.git
   cd fuzzy-suction-optimization

2. Install dependencies
   ```bash
   pip install -r requirements.txt

3. Run the program
   ```bash
   python vacuum_fuzzy.py

4. Enter input values when prompted

    • Floor surface condition (1–100)

    • Dust level (1–100)

## Limitations
- Input values are assumed to be ideal sensor readings
- The system is currently evaluated in an offline simulation environment
- Hardware integration and real-time sensing are not included
