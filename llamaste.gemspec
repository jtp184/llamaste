# frozen_string_literal: true

require_relative 'lib/llamaste/version'

Gem::Specification.new do |spec|
  spec.name = 'llamaste'
  spec.version = Llamaste::VERSION
  spec.authors = ['Justin Piotroski']
  spec.email = ['jtp184@gmail.com']

  spec.summary = 'Llamaste: the Simply Translated Extension for LLaMa'
  spec.homepage = 'https://github.com/jtp184/llamaste'
  spec.required_ruby_version = '>= 3.2.0'

  spec.metadata['homepage_uri'] = spec.homepage

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) || f.start_with?(*%w[bin/ test/ spec/ features/ .git .circleci appveyor])
    end
  end

  spec.add_dependency 'matrix', '0.4.2'

  spec.bindir = 'exe'
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.extensions = ['ext/ruby_llama/extconf.rb']
  spec.require_paths = ['lib']
end
