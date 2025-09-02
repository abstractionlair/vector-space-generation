# Vector-Space Generation

Exploring language model generation in continuous vector space before token discretization.

## Hypothesis

Language models have learned to understand a continuous "language" where every point in their embedding space represents a valid semantic unit. Standard generation forces discretization at each step (vector → token → vector), potentially losing information. This project tests whether generating multiple vectors before discretization preserves information and affects generation properties.

## Approach

Instead of the standard token-by-token generation:
1. Generate N vectors in sequence without discretizing to tokens
2. After N steps, translate the vector sequence back to natural language
3. Compare with standard generation

## Project Structure

```
vector-space-generation/
├── experiments/        # Experiment scripts and notebooks
├── src/               # Core implementation
├── results/           # Experimental results and analysis
└── docs/              # Documentation and notes
```

## Setup

```bash
# Clone repository
git clone https://github.com/abstractionlair/vector-space-generation.git
cd vector-space-generation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers numpy matplotlib jupyter
```

## Status

Currently in initial development - setting up baseline implementation and testing infrastructure.

## License

MIT
