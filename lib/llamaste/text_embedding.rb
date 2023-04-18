# frozen_string_literal: true

require 'matrix'
require 'digest'

module Llamaste
  # Represents the Embedding vectors
  class TextEmbedding
    extend Forwardable
    include Comparable

    # Original string input
    attr_reader :string
    # Array of embedding vectors
    attr_reader :embeddings
    # Matrix of embeddings
    attr_reader :vector

    def_delegators :to_a, :each, :count

    # Take in values for +string+ and +tokens+
    def initialize(string, embeds)
      @string = string
      @embeddings = embeds
      @vector = Matrix[[*embeds]]
    end

    # Return copy of +embeddings+
    def to_a
      embeddings.dup
    end

    # Return copy of +string+
    def to_s
      string.dup
    end

    # Compares +other+ to self using cosine similarity
    def <=>(other)
      return unless other.is_a?(self.class)

      cosine_similarity(vector, other.vector)
    end

    # Returns a digest for the embedding, using the +algo+ and +format+ to pick a digest subclass
    def digest(algo = :SHA1, format = :hex)
      Digest.const_get(algo)
            .send(:"#{format}digest", to_a.unshift(to_s).join)
    end

    alias to_ary to_a
    alias to_str to_s

    private

    # returns the dot product of +left+ and +right+ matrixes
    def dot_product(left, right)
      (left.transpose * right).element(0, 0)
    end

    # returns the magnitude of a +vector+
    def magnitude(vector)
      Math.sqrt(dot_product(vector, vector))
    end

    # returns cosine similarity for +left+ and +right+ vectors
    def cosine_similarity(left, right)
      dot = dot_product(left, right)
      x = magnitude(left)
      y = magnitude(right)

      return 0 if [x, y].any?(&:zero?)

      dot / (x * y)
    end
  end
end
