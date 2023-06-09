# frozen_string_literal: true

require 'bundler/gem_tasks'
require 'rspec/core/rake_task'
require 'rake/extensiontask'

RSpec::Core::RakeTask.new(:spec)

require 'rubocop/rake_task'

RuboCop::RakeTask.new

task default: %i[spec rubocop bump docs compile]

Rake::ExtensionTask.new('ruby_llama') do |ext|
  ext.lib_dir = 'lib/ruby_llama'
end

RDOC_EXCLUDE = %w[
  bin/setup
  bin/console
  bin/basic
  coverage
  Gemfile
  Gemfile.lock
  Rakefile
  tmp
  docs
  spec
  ext/ruby_llama
].map { |r| "--exclude=#{r}" }.join(' ').freeze

task :docs do
  sh "rdoc --output=docs --format=hanna --all --main=README.md #{RDOC_EXCLUDE}"
end

task :docs? do
  sh "rdoc -C --all #{RDOC_EXCLUDE}"
end

task :bump do
  version_file = './lib/llamaste/version.rb'
  matcher = /VERSION = '(.*)'/

  file_contents = File.read(version_file)

  updated = file_contents.gsub(matcher) do |v|
    v.gsub Regexp.last_match(1), Regexp.last_match(1).succ
  end

  File.open(version_file, 'w+') { |f| f << updated }

  sh 'bundle'
end
