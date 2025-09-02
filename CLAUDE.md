# Claude Context for Vector-Space Generation Project

## Project Overview
We are testing whether language models can generate more effectively by operating in continuous vector space for multiple steps before discretizing to tokens. The hypothesis is that models understand a continuous "language" where every vector in embedding space is potentially meaningful.

## Core Technical Approach
- Generate N vectors sequentially without token discretization (using concept tokens or raw states)
- Translate vector sequences back to text via: `Embedded("Please translate the following: ")` + `vector_sequence`
- Test interleaved pattern: generate → translate → continue generating
- Initial experiments WITHOUT additional context to test self-containment

## Key Decisions Made
1. **Model**: Starting with GPT-2 small (124M params) on M4 MacBook Pro
2. **Initial Test**: N=2-4 vectors before discretization (with entropy-based early stop)
3. **Two methods**: Concept tokens (stable) and raw hidden states (experimental)
4. **Development**: Local on M4 with MPS acceleration
5. **Translation**: "Please translate the following:" without original context

## Novel Contributions
1. **Interleaved generation**: Alternate between continuous and discrete modes
2. **Explicit translation**: Treat vector sequences as messages to interpret
3. **Systematic N study**: Test how checkpoint frequency affects quality
4. **Continuation after translation**: Resume from discrete checkpoints

## Building On
- **Soft Thinking**: Using their concept token approach for stability
- **COCONUT**: Different from their abstract reasoning approach
- **CoT2**: Related theoretical framework

## Project Philosophy
- This is empirical research - we don't know if it will work
- Negative results are valuable information
- Stay focused on testing the core hypothesis first
- Avoid premature optimization or feature expansion

## Current Status
- Repository created at https://github.com/abstractionlair/vector-space-generation
- Specifications complete
- Ready for implementation with Claude Code

## Communication Style
- Direct technical discussion without excessive enthusiasm
- Acknowledge uncertainties and potential failure modes
- Focus on empirical testing over theoretical speculation
- Treat this as research, not advocacy for an approach

## Open Questions
1. How to handle EOS token in vector space?
2. Will raw states work with small N or need aligner?
3. What's the optimal interleaving rhythm?
4. How does entropy gating interact with translation quality?

## Future Extensions (if initial tests succeed)
- Test with larger models
- Try vector-space Chain-of-Thought reasoning
- Consider RL training to improve vector generation
- Cross-model translation tests

## Remember
The hypothesis might be wrong. The approach might not work with pretrained models. That's fine - we're testing an idea, not proving a predetermined conclusion.