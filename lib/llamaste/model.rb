# frozen_string_literal: true

module Llamaste
  # Defines model path and parameters, hybrid C model
  class Model
    # Configurable params for generator
    attr_accessor :params

    class << self
      def load(**params)
        new(**params).tap(&:load)
      end
    end

    # Takes in +params+ and saves
    def initialize(**kwargs)
      @params = kwargs
    end

    # Passes the params to the C load_model method
    def load
      @load ||= load_model(params)
    end

    # Force a load even if we've loaded already
    def load!
      @load = nil
      load
    end

    def loaded?
      @load
    end

    def tokenize(input)
      TokenSequence.new(input, tokenize_text(input))
    end
  end
end
