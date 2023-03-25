# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Llamaste::TokenEmbedding do
  let(:string) { 'Some Text' }
  let(:tokens) { Array.new(3) { ['', rand(10_000)] } }

  subject { described_class.new(string, tokens) }

  describe '#to_a' do
    it 'returns the token integers' do
      expect(subject.to_a).to be_a(Array)
      expect(subject.to_a).to all(be_a(Integer))
    end
  end

  describe '#to_s' do
    it 'returns the string' do
      expect(subject.to_s).to eq(string)
    end
  end
end
