# frozen_string_literal: true

module Llamaste
  # Hold the original string and its represnentative tokens
  class TokenGroup
    extend Forwardable

    # Original string input
    attr_reader :string
    # Array of token ids
    attr_reader :tokens

    def_delegators :to_a, :each, :count

    # Take in values for +string+ and +tokens+
    def initialize(string, tokens)
      @string = string
      @tokens = tokens
    end

    # Return copy of +ids+
    def to_a
      tokens.dup.map(&:last)
    end

    # Return copy of +string+
    def to_s
      string.dup
    end

    alias to_ary to_a
    alias to_str to_s
  end
end
