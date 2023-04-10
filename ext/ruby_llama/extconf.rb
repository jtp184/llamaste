# frozen_string_literal: true

require 'mkmf'

dir_config('my_extension', '.', '.')
create_makefile 'llamaste/ruby_llama'
