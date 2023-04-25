# frozen_string_literal: true

require 'etc'

module Llamaste
  # Core model which wraps the text generation functions
  class Model
    # Filepath of the model being used
    attr_accessor :model
    # Configurable params for generator
    attr_accessor :params
    # Text result of generation session
    attr_reader :output

    # Takes in +params+, extracts out +model+, and saves merging in +default_params+
    def initialize(params = {})
      @model = params.delete(:model)
      @params = self.class.default_params.merge(params)
    end

    # Default parameters for model
    def self.default_params
      {
        seed: Time.now.to_i,
        threads: Etc.nprocessors,
        tokens: 128,
        penalize_tokens: 64,
        parts: 1,
        context_size: 512,
        top_k: 40,
        top_p: 0.95,
        temperature: 0.8,
        repeat_penalty: 1.1,
        batch_size: 8,
        memory_lock: false,
        memory_f16: false,
        use_mmap: true,
        lora_base: nil,
        lora_adapter: nil,
        embedding: false
      }
    end

    # Loads the model into memory, setting a new +model_path+ if given
    def load_model(model_path = nil)
      if model_path
        @model = model_path
        load_model
      elsif model
        @llama = nil
        llama.load_model
      else
        raise KeyError, 'No model provided'
      end

      model
    end

    # Return a TokenGroup for +input+ string
    def tokenize(input)
      TokenGroup.new(input, llama.tokenize_text(input))
    end

    # Returns a TextEmbedding for +input+ string. Raises a KeyError if not in embedding mode
    def embed(input)
      raise KeyError, 'Need to set embedding mode' unless params[:embedding]

      TextEmbedding.new(input, llama.embed_text(input))
    end

    # Returns a binary blob representing the model state after processing the +input_prompt+
    def cache_prompt(input_prompt)
      llama.cache_prompt(parse_input(input_prompt))
    end

    # Loads the +state_data+ binary blob as the model state, and resumes token processing for the +input_prompt+
    # with the same +break_on+ options as #call
    def resume_prompt(input_prompt, state_data, break_on: nil)
      llama.resume_prompt(parse_input(input_prompt), state_data, break_on)
    end

    # Runs quantization on the +input_file+ defaulting to the model, emitting it to
    # +output_file+ using the +quantize_type+
    def quantize(input_file: model, output_file: nil, quantize_type: :q4_0)
      output_file ||= input_file.gsub(/-[^-]+(?=\.bin$)/, "-#{quantize_type}")
      itype = {
        f16: 1,
        q4_0: 2,
        q4_1: 3,
        q4_1a: 4,
        q4_2: 5,
        q4_3: 6
      }[quantize_type]

      llama.quantize(input_file, output_file, itype)
    end

    # Takes in a TokenGroup, string-like or array-like +input+ and generates text.
    # Stops early if it encounters one of the strings in an array passed to +break_on+
    # Returns the output, and yields it piece by piece
    def call(input, break_on: nil, &blk)
      @output = llama.process_tokens(parse_input(input), break_on, &blk)
    end

    private

    # Takes in the +input_prompt+ and optionally prepends a +bos+ token
    def parse_input(input_prompt, bos: true)
      tkns = case input_prompt
             when ->(i) { i.respond_to?(:to_ary) } then input_prompt
             when ->(i) { i.respond_to?(:to_str) } then tokenize(input_prompt)
             end.to_ary.dup

      tkns.unshift(1) if bos
      tkns
    end

    # The wrapped C GPT transformer
    def llama
      @llama ||= Llama.new(params.merge(model:))
    end
  end
end
