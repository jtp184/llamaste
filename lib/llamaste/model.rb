# frozen_string_literal: true

module Llamaste
  # Defines model path and parameters, hybrid C model
  class Model
    # Path to model on disk
    attr_reader :model_path
    # Use mmap if possible
    attr_reader :use_mmap
    # Force system to keep model in RAM
    attr_reader :use_mlock

    class << self
      def load(**params)
        new(**params).tap(&:load)
      end
    end

    def initialize(model_path:, use_mmap: true, use_mlock: false)
      @model_path = model_path
      @use_mmap = use_mmap
      @use_mlock = use_mlock
    end

    # Passes the params to the C load_model method
    def load
      @load ||= load_model
    end

    # Force a load even if we've loaded already
    def load!
      @load = nil
      load
    end

    # Check if the model has been loaded
    def loaded?
      @load
    end
  end
end
