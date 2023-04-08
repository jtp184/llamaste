# Llamaste


Llamaste is a Ruby C Extension which wraps [llama.cpp](https://github.com/ggerganov/llama.cpp) to enable using the LLaMA text model inside Ruby.

## Installation

Add to your gemfile using Bundler

```ruby
gem 'llamaste', :git => 'git://github.com/jtp184/llamaste.git'
```

or clone and install globally using `rake compile install`

## Usage

### Quickstart

```ruby
# Configure model params like filepath, tokens to generate, context size, etc
# These are the defaults (apart from model)
params = {
  model: './models/30B/ggml-model-q4_0.bin',
  seed: Time.now.to_i,
  threads: Etc.nprocessors, # n_threads
  tokens: 128, # n_predict
  penalize_tokens: 64, # repeat_last_n
  parts: 1, # n_parts
  context_size: 512, # n_ctx
  top_k: 40,
  top_p: 0.95,
  temperature: 0.8, # temp
  repeat_penalty: 1.1,
  batch_size: 8, # n_batch
  memory_lock: false, # use_mlock
  memory_f16: false
}

@model = Llamaste::Model.new(params)

# Load model into memory
@model.load_model

text_input = 'It was a dark and stormy night'

# Tokenize text

token = @model.tokenize(text_input)
# => 
# <Llamaste::TokenEmbedding:0x00007f564a6a9b48                                  
#  @string="It was a dark and stormy night",                                     
#  @tokens=                                                                      
#   [["It", 3112],                                                               
#    [" was", 471],                                                              
#    [" a", 263],                                                                
#    [" dark", 6501],                                                            
#    [" and", 322],                                                              
#    [" storm", 14280],                                                          
#    ["y", 29891],                                                               
#    [" night", 4646]]
# >

# Generate based on token or text, returns a string.
# Providing a block will yield a string for each generated token

@model.call(token) { |tkn| print tkn }
# => ", and I was on a plane headed for somewhere, but I didn’t know where."
```

## Contributing

Bug reports, feature interest, and pull requests are welcome on GitHub at https://github.com/jtp184/llamaste.

### Project Goals

- Remain up to date with `llama.cpp`
- Increase feature offering, configurability
- Increase quality of extension C++
- Simple ChatGPT clone running on local via rails

## References and Acknowledgements

- [Georgi Gerganov's](https://github.com/ggerganov) amazing [llama.cpp](https://github.com/ggerganov/llama.cpp) code, this gem would not be possible without it
- Meta's [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) model
- [The Ruby C API](http://silverhammermba.github.io/emberb/c/#data)

 ---
![image of a llama who is a monk](https://github.com/jtp184/llamaste/blob/main/dream-llamas.jpg?raw=true)
