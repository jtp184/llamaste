# frozen_string_literal: true

require 'mkmf'

$CFLAGS += ' ' + %w[-Icommon -D_GNU_SOURCE  -DGGML_USE_LLAMAFILE -std=c11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration -pthread -march=native -mtune=native -fopenmp -Wdouble-promotion].join(' ')
$CXXFLAGS += ' ' + %w[-std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -fopenmp -march=native -mtune=native -Wno-array-bounds -Wno-format-truncation -Wextra-semi -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_OPENMP -DGGML_USE_LLAMAFILE].join(' ')

dir_config('ruby_llama', '.', '.')
create_makefile 'llamaste/ruby_llama'
