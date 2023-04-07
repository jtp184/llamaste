# frozen_string_literal: true

require 'spec_helper'

RSpec.describe Llamaste::Model do
  let(:llama) do
    double(
      load_model: nil,
      tokenize_text: [['', 0]],
      process_text: '',
      process_tokens: ''
    )
  end

  let(:model_path) { '/models/model-b1.bin' }
  subject(:service) { described_class.new }

  before do
    allow_any_instance_of(Llamaste::Model).to receive(:llama).and_return(llama)
  end

  describe '#initialize' do
    it 'sets the model when given one' do
      expect(described_class.new(model: model_path).model).to eq(model_path)
    end

    it 'sets default model parameters when not set' do
      %i[top_k threads temperature tokens].each do |param|
        expect(described_class.new.params).to have_key(param)
      end
    end

    it 'overrides model parameters when set' do
      expect(described_class.new(top_k: 4000).params[:top_k]).to eq(4000)
    end
  end

  describe '#load_model' do
    context 'when given a path string' do
      it 'first sets the model path to that string' do
        expect { service.load_model(model_path) }.to(
          change { service.model }.from(nil).to(model_path)
        )
      end
    end

    context 'when no model path exists' do
      it 'raises a KeyError' do
        expect { service.load_model }.to raise_error(KeyError)
      end
    end

    context 'when model path is given' do
      let(:service) { described_class.new(model: model_path) }

      it 'loads the model from the filesystem' do
        expect(llama).to receive(:load_model)
        service.load_model
      end
    end
  end

  describe '#tokenize' do
    it 'returns a TokenEmbedding' do
      expect(service.tokenize('Some Text')).to be_a(Llamaste::TokenEmbedding)
    end
  end

  describe '#quantize' do
    let(:service) { described_class.new(model: model_path) }
    let(:modified_path) { '/models/model-q4_0.bin' }

    it 'quantizes the provided model' do
      expect(llama).to receive(:quantize).with(
        model_path,
        modified_path,
        2
      )

      service.quantize
    end
  end

  describe '#call' do
    subject { service.call(input) }

    context 'when given a string-like' do
      let(:input) { 'Some Text' }

      it 'calls process_text' do
        expect(llama).to receive(:process_text)
        subject
      end
    end

    context 'when given a TokenEmbedding' do
      let(:input) { Llamaste::TokenEmbedding.new('It was a dark and stormy night', Array.new(7, [])) }

      it 'calls process_tokens' do
        expect(llama).to receive(:process_tokens)
        subject
      end
    end

    context 'when given an array-like' do
      let(:input) { (1..20).to_a.shuffle }

      it 'calls process_tokens' do
        expect(llama).to receive(:process_tokens)
        subject
      end
    end
  end
end
