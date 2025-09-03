# Continuous Vector Space Internal Language for LLMs

## Conjecture

While studying LLM architecture and having been exposed to examples like the King - Man + Woman = Queen example using embedding vectors, I wondered if models would understand vectors that didn't correspond to words/tokens in the target language. (And I expect many people have as well.) In particular, I wondered if existing models _already_ understood and used such an expanded language internally, expecially in middle layers, but might be hobbled by being forced to choose a target language token at the end of each pass. In the examople, if we imagined that English didn't have the word "Queen", it would make sense for it to be one, and it might be that models would work as-if it was one. At least until some final translation step to produce output the user could understand. I decided I could test this.

## Test

### Plan 1

I planned to take a small model and modify it so that we would have a writer mode with the following changes.
1. We could take over the embeding step and feed the model sequences of vectors.
2. We could take the raw hidden vector from just before choosing a next token and make that the next vector.

And then to implement a reader mode with the following changes.
1. As above, we could take over the embeding step and feed the model sequences of vectors.
2. It _would_ be forced to pick a next token at each step as usual, no raw vectors.
We would then send a query to the writer model, collect the output as a sequence of vectors, and then call the reader model with a message like Embedded("Please translate the following into English: ") + (Sequence of Vectors).

#### Issues

I ran this by several models and got the most constructive feedback from GPT-5 Pro.

A lesser problem was that it predicted that while this might work a bit at first, but as the vector output sequence grew it expected it to get so far from the distribution seen in training and for output to become garbage. I call this the lesser problem because that would still be output and a result, even if a negative one.

The larger problem was that the vectors would have positions from the writer encoded in them which could mess with the reader's attention mechanism. It also pointed to some existing research, which I will cite later. (I should have done a literature search earlier. I am a bit surprised other models hadn't brought this up.) There were a couple of ways to go here. One was based on "Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space" which includes the idea of concept tokens. By superimposing the embeddings for the next tokens the model would have chose from to create the next vector ourselves, we naturally strip away things like the position endcodings _and_ we force ourselves to stay in the area of the space populated by token embeddings, while still allowing us to stray from the target language. There were other options, but I wanted to choose one manageable thing and try it so I went with this.

I also decided to include an idea of limiting the number of vectors produced before forcing a translation to the target language and then continuuing generation in the writer. This would lead to an interleaving of next vector chunk generation and chunk translation. This would be another mechanism to keep the process from wandering out of the well-supported part of the space.

This led to

### Plan 2

We would have a writer mode with the following changes.
1. We could take over the embeding step and feed the model sequences of vectors.
2. We could take the raw hidden vector from just before choosing a next token.
3. We would proceed to generating the next-token probability distribution, and create a next vector as a probability-weighted sum of the embeddings of the next tokens. (This is an average, but it doesn't feel like an average. I may just need time to let it sink in.) And this would be the next vector
4. If we hit the limit on the number of vectors produced, trigger a call to the reader to translate the pending vectors and then replace them in our (vector) context with the embeddings.

And then to implement a reader mode with the following changes. 
1. As above, we could take over the embeding step and feed the model sequences of vectors.
2. It _would_ be forced to pick a next token at each step as usual, no raw vectors.
We would then send a query to the writer model, collect the output as a sequence of vectors, and then call the reader model with a message like Embedded("Please translate the following into English: ") + (Sequence of Vectors).

#### Implementation 1
The initial implementation used GPT-2. This ran into two problems. First, I hand't considered a comoletion model rather than an instruction model. We changed the translation step to "The meaning of (Sequence of Vectors) in plain English is ". But then we ran into the second problem. Even without our weird vectors, we couldn't get GPT-2 do translations like this, even for Spanish to English.

#### Implementation 2
We switched to Qwen2.5-1.5B and attempted to go back to instruction mode. We could not get this to work, we believe because in this mode the model needed to see "<|im_start|>" and "<|im_end|>" explicitly as _tokens_. Trying to put the equivalent in the sequence of vectors didn't work. So we fell back to using completion mode. Good news here was that the Spanish to English translation done in completion style did work with this model. With this implementation we did get results.

## Results
Words did come out! Though not always in English and not in meaningful responses. However, things became clearer when looking at the nearest tokens to the generated vectors. It became apparent that they did not correspond to a single answer, just in a generalized language. They were superpositions of truly different potential responses. For instance, "The capital of France is" -> ("Paris", "located", "a") which you could imagine being completed as "The capital of France is Paris", "The capital of France is located on a river.", and "The capital of France is a popular vacation destination.".

This does lend weight to the ideas in "Training Large Language Models to Reason in a Continuous Latent Space" in which this can be used to explore multiple approaches in CoT at once. 

## Loose Ends
A few other things were considered and/or implemented but didn't end up being very relevant since the conclusion was apparent from just the first couple of vectors. GPT-5 Pro pointed out that we'd need some new mechanism for deciding when to stop generating vectors since our implementation would break the mechanism based on a stop-token and it suggested monitoring entropy. (This was not an original idea, it is in the reseasrch GPT-5 Pro pointed to.) We did this and in the GPT-2-based tests it needed some tuning to avoid going on too long. The approach also worked for the Qwen implementation. The interleaved generation of chunks and translation of chunks didn't matter for similar reasons.

While my conjecture was that this could be done without any special model training, I did discuss with Claude how to try to train a model so that this would work if we had negative results. The idea here was to train a model the usual way, both pre and post, then to use RL with the model generating a max of two vectors at a time before translation, then three at a time, ... 

## Existing Research

### Training Large Language Models to Reason in a Continuous Latent Space
https://arxiv.org/abs/2412.06769

### Continuous Chain of Thought Enables Parallel Exploration and Reasoning
https://arxiv.org/abs/2505.23648

### Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space
https://arxiv.org/abs/2505.15778v1

## Conclusion

Knowing what I do now, this could have been seen by _just_ adding some logging of the distribution of next tokens before generating them but otherwise letting the model run as normal.

