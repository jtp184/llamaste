# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Llamaste::TextEmbedding do
  let(:string) { 'Some Text' }
  let(:embeddings) { Array.new(10) { rand(-1.0..1.0) } }

  subject(:embedding) { described_class.new(string, embeddings) }

  describe '#to_a' do
    it 'returns the embedding floats' do
      expect(subject.to_a).to be_a(Array)
      expect(subject.to_a).to all(be_a(Float))
    end
  end

  describe '#to_s' do
    it 'returns the string' do
      expect(subject.to_s).to eq(string)
    end
  end

  describe '#<=>' do
    let(:other) do
      described_class.new('Other Text', Array.new(10) { rand(-1.0..1.0) })
    end

    subject { embedding.<=>(other) }

    it 'returns a comparitor based on cosine similarity' do
      expect(embedding).to receive(:cosine_similarity).and_call_original
      expect(subject).to be_a(Float)
    end
  end
end
