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
        memory_f16: false
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

    # Return a TokenEmbedding for +input+ string
    def tokenize(input)
      TokenEmbedding.new(input, llama.tokenize_text(input))
    end

    # Runs quantization on the +input_file+ defaulting to the model, emitting it to
    # +output_file+ using the +quantize_type+
    def quantize(input_file: model, output_file: nil, quantize_type: :q4_0)
      output_file ||= input_file.gsub(/-[^-]+(?=\.bin$)/, "-#{quantize_type}")
      itype = case quantize_type
              when :q4_0 then 2
              when :q4_1 then 3
              end

      llama.quantize(input_file, output_file, itype)
    end

    # Takes in a TokenEmbedding, string-like or array-like +input+ and generates text.
    # Stops early if it encounters one of the strings in an array passed to +break_on+
    # Returns the output, and yields it piece by piece
    def call(input, break_on: nil, &blk)
      @output = case input
                when TokenEmbedding then handle_ary_input(input, break_on, &blk)
                when ->(i) { i.respond_to?(:to_str) } then handle_str_input(input, break_on, &blk)
                when ->(i) { i.respond_to?(:to_ary) } then handle_ary_input(input, break_on, &blk)
                end
    end

    private

    # Takes in a string-like +input+, tokenizing it if we are caching tokens, and processing it as text if we are not
    def handle_str_input(input, break_on, &blk)
      llama.process_text(input, break_on, &blk)
    end

    # Takes an arrayable +input+ and processes it as tokens, adding a BOS header to it
    def handle_ary_input(input, break_on, &blk)
      bos = input.to_ary.dup.unshift(1)

      llama.process_tokens(bos, break_on, &blk)
    end

    # The wrapped C GPT transformer
    def llama
      @llama ||= Llama.new(params.merge(model:))
    end
  end
end
