# Claude Context for Vector-Space Generation Project

## Project Overview
We are testing whether language models can generate more effectively by operating in continuous vector space for multiple steps before discretizing to tokens. The hypothesis is that models understand a continuous "language" where every vector in embedding space is potentially meaningful.

## Core Technical Approach
- Generate N vectors sequentially without token discretization
- Translate vector sequences back to text via: `Embedded("Please translate the following: ")` + `vector_sequence`
- Compare with standard token-by-token generation

## Key Decisions Made
1. **Model**: Starting with GPT-2 small (124M params) on M4 MacBook Pro
2. **Initial Test**: N=2 vectors before discretization
3. **Vector Transition**: Use pre-final-layer hidden states directly
4. **Development**: Local on M4 with MPS acceleration

## Project Philosophy
- This is empirical research - we don't know if it will work
- Negative results are valuable information
- Stay focused on testing the core hypothesis first
- Avoid premature optimization or feature expansion

## Current Status
- Repository created at https://github.com/abstractionlair/vector-space-generation
- Setting up local development environment
- Next: Create baseline GPT-2 generation script

## Communication Style
- Direct technical discussion without excessive enthusiasm
- Acknowledge uncertainties and potential failure modes
- Focus on empirical testing over theoretical speculation
- Treat this as research, not advocacy for an approach

## Open Questions
1. How to handle EOS token in vector space?
2. Should N vary based on task complexity?
3. Are there natural "chunk boundaries" in vector space?
4. How does causal masking interact with multi-vector generation?

## Future Extensions (if initial tests succeed)
- Vary N dynamically
- Test on more complex tasks
- Try vector-space Chain-of-Thought reasoning
- Consider RL training to improve vector generation

## File Structure Plan
```
vector-space-generation/
├── experiments/        # Test scripts
├── src/               # Core implementation
├── results/           # Outputs and analysis
├── docs/              # Documentation
└── notebooks/         # Interactive exploration
```

## Remember
The hypothesis might be wrong. The approach might not work with pretrained models. That's fine - we're testing an idea, not proving a predetermined conclusion.
