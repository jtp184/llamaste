#include "llama.h"
#include "common.h"
#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <codecvt>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numa.h>
#include <pthread.h>
#include <regex>
#include <sched.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ruby.h"
#include <ruby/thread.h>

// COMMON

static VALUE tokens_to_rb_array(struct llama_context * ctx, const std::vector<llama_token>& tokens) {
  VALUE ruby_array = rb_ary_new2(tokens.size());

  for(const auto& token : tokens) {
      VALUE tuple = rb_ary_new2(2);

      rb_ary_push(tuple, rb_str_new_cstr(llama_token_to_piece(ctx, token).c_str()));
        rb_ary_push(tuple, INT2NUM(token));
        rb_ary_push(ruby_array, tuple);
  }

  return ruby_array;
}

// MODEL

struct model_data { 
  llama_context *ctx;
  std::string model_path;
  llama_model *model;
  llama_model_params m_params;
  llama_context_params c_params;
};

static const rb_data_type_t model_type = {
  .wrap_struct_name = "ModelData",
  .function = {
    .dmark = NULL,
    .dfree = RUBY_DEFAULT_FREE,
    .dsize = [](const void *p) { return sizeof(model_data); }
  },
  .data = NULL,
  .flags = RUBY_TYPED_FREE_IMMEDIATELY
};

static VALUE m_tokenize_text(VALUE self, VALUE input_str) {
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  auto tokens_list = ::llama_tokenize(data->ctx, StringValueCStr(input_str), true);

  return tokens_to_rb_array(data->ctx, tokens_list);
}

static int m_parse_params_iter_callback(VALUE key, VALUE value, VALUE self) {
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  std::string key_str = rb_id2name(SYM2ID(key));
  
  if(key_str == "seed") { data->c_params.seed = NUM2INT(value); }
  else if(key_str == "model_path") { data->model_path = StringValueCStr(value); }
  else if(key_str == "threads" || key_str == "n_threads") { data->c_params.n_threads = NUM2INT(value); }
  else if(key_str == "use_mmap") { data->m_params.use_mmap = RTEST(value); }
  else if(key_str == "use_mlock") { data->m_params.use_mlock = RTEST(value); }

  return ST_CONTINUE;
}

static void m_apply_params(VALUE self, VALUE params_hash) {
  rb_hash_foreach(
    params_hash,
    m_parse_params_iter_callback,
    self
  );
}

static VALUE m_start_backend(int argc, VALUE *argv, VALUE self) {
  int useNuma = 0;
  if(argc == 1) { useNuma = NUM2INT(argv[0]); }

  llama_backend_init();
  llama_numa_init((ggml_numa_strategy)useNuma);

  return Qtrue;
}

static VALUE m_stop_backend(VALUE self) {
  llama_backend_free();

  return Qtrue;
}

static VALUE m_load_model(VALUE self, VALUE params_hash) {
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  data->m_params = llama_model_default_params();
  data->c_params = llama_context_default_params();

  m_apply_params(self, params_hash);

  data->model = llama_load_model_from_file(
    data->model_path.c_str(),
    data->m_params
  );

  if (data->model == NULL) { return Qfalse; }

  data->ctx = llama_new_context_with_model(
    data->model,
    data->c_params
  );

  if (data->ctx == NULL) { return Qfalse; }

  return Qtrue;
}

static VALUE m_allocate(VALUE klass) {
  model_data *data = (model_data*)ruby_xmalloc(sizeof(model_data));

  data->model = NULL;
  data->ctx = NULL;

  return TypedData_Wrap_Struct(klass, &model_type, data);
}

static VALUE m_close(VALUE self) {
  /* fprintf(stderr, "Closing model\n"); */
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  llama_free(data->ctx);
  data->ctx = NULL;
  llama_free_model(data->model);
  data->model = NULL;
  ruby_xfree(data);

  return Qnil;
}

// INFERENCE

struct inference_data {
  std::string prompt;
  std::vector<llama_token> tokens;
  llama_batch batch;
  int n_predict;
  bool callback_block;
  std::string output_str;
  std::vector<llama_token> output_tokens;
};

struct inference_context {
  model_data *m_data;
  inference_data *i_data;
};

static const rb_data_type_t inference_type = {
  .wrap_struct_name = "InferenceData",
  .function = {
    .dmark = NULL,
    .dfree = RUBY_DEFAULT_FREE,
    .dsize = [](const void *p) { return sizeof(inference_data); }
  },
  .data = NULL,
  .flags = RUBY_TYPED_FREE_IMMEDIATELY
};

static void *i_resync_execute_block(void *data) {
  VALUE *str = (VALUE *)data;
  rb_yield(*str);
}

static std::string i_predict_tokens(model_data *m_data, inference_data *i_data) {
  // Buffer for output
  std::ostringstream output_buffer;

  // Create initial tokens
  i_data->tokens = ::llama_tokenize(
      m_data->ctx,
      i_data->prompt.c_str(),
      true
  );

  // TODO: Does this batch size matter?
  i_data->batch = llama_batch_init(512, 0, 1);

  for(size_t i = 0; i < i_data->tokens.size(); i++) {
    llama_batch_add(i_data->batch, i_data->tokens[i], i, { 0 }, false);
  }

  // llama_decode will output logits only for the last token of the prompt
  // TODO: Play with this
  i_data->batch.logits[i_data->batch.n_tokens - 1] = true;

  // Decode initial prompt
  llama_decode(m_data->ctx, i_data->batch);

  // Counters
  int n_cur    = i_data->batch.n_tokens;
  int n_decode = 0;

  // Main loop
  while (n_cur <= i_data->n_predict) {
    {
      auto n_vocab = llama_n_vocab(m_data->model);
      auto *logits = llama_get_logits_ith(m_data->ctx, i_data->batch.n_tokens - 1);

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);

      for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
      }

      llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

      // sample the most likely token
      const llama_token new_token_id = llama_sample_token_greedy(m_data->ctx, &candidates_p);

      // Add token to outputs
      std::string token = llama_token_to_piece(m_data->ctx, new_token_id);
      output_buffer << token;
      i_data->output_tokens.push_back(new_token_id);

      // Call callback block if it exists
      if(i_data->callback_block) {
        VALUE rb_str = rb_str_new_cstr(token.c_str());

        rb_thread_call_with_gvl(
          i_resync_execute_block,
          (void *)&rb_str
        );
      }

      // is it an end of generation?
      if (llama_token_is_eog(m_data->model, new_token_id) || n_cur == i_data->n_predict) {
        break;
      }

      // prepare the next batch
      llama_batch_clear(i_data->batch);

      // add next token to eval
      llama_batch_add(i_data->batch, new_token_id, n_cur, { 0 }, true);

      n_decode += 1;
    }

    n_cur += 1;

    // Evaluate current batch with transformer model
    llama_decode(m_data->ctx, i_data->batch);
  }

  // Return final output
  std::string output = output_buffer.str();
  return output;
}

static void *i_async_process_tokens(void *data) {
  inference_context *i_context = (inference_context *)data;

  i_context->i_data->output_str = i_predict_tokens(
    i_context->m_data,
    i_context->i_data
  );

  return NULL;
}
static VALUE i_process_tokens(VALUE self, VALUE model, VALUE prompt, VALUE predict) {
  inference_data *i_data;
  TypedData_Get_Struct(self, inference_data, &inference_type, i_data);

  model_data *m_data;
  TypedData_Get_Struct(model, model_data, &model_type, m_data);

  i_data->prompt = StringValueCStr(prompt);
  i_data->callback_block = rb_block_given_p();
  i_data->n_predict = NUM2INT(predict);

  inference_context i_context;
  i_context.m_data = m_data;
  i_context.i_data = i_data;

  rb_thread_call_without_gvl(
    i_async_process_tokens,
    (void *)&i_context,
    RUBY_UBF_IO,
    0
  );

  // Return string and tokens
  VALUE tuple = rb_ary_new2(2);
  rb_ary_push(tuple, rb_str_new_cstr(i_data->output_str.c_str()));
  rb_ary_push(tuple, tokens_to_rb_array(m_data->ctx, i_data->output_tokens));

  return tuple;
}

static VALUE i_close(VALUE self) {
  /* fprintf(stderr, "Closing inference\n"); */
  inference_data *data ;
  TypedData_Get_Struct(self, inference_data, &inference_type, data);

  data->prompt.~basic_string();
  data->tokens.~vector();
  data->output_str.~basic_string();
  data->output_tokens.~vector();

  llama_batch_free(data->batch);
  ruby_xfree(data);

  return Qnil;
}

static VALUE i_allocate(VALUE klass) {
  inference_data *data = static_cast<inference_data*>(ruby_xmalloc(sizeof(inference_data)));
  
  new (&data->prompt) std::string();
  new (&data->tokens) std::vector<llama_token>();
  new (&data->output_str) std::string();
  new (&data->output_tokens) std::vector<llama_token>();
  new (&data->batch) llama_batch();

  data->n_predict = 0;
  data->callback_block = false;

  return TypedData_Wrap_Struct(klass, &inference_type, data);
}

// DEFINITIONS

extern "C" void Init_ruby_llama() {
  VALUE mLlamaste = rb_define_module("Llamaste");

  // MODEL
  VALUE cModel = rb_define_class_under(mLlamaste, "Model", rb_cObject);
  rb_define_alloc_func(cModel, m_allocate);
  rb_define_method(cModel, "close", (VALUE(*)(ANYARGS))m_close, 0);
  rb_define_method(cModel, "load_model", (VALUE(*)(ANYARGS))m_load_model, 1);
  rb_define_method(cModel, "tokenize_text", (VALUE(*)(ANYARGS))m_tokenize_text, 1);
  rb_define_singleton_method(cModel, "start_backend", m_start_backend, -1);
  rb_define_singleton_method(cModel, "stop_backend", m_stop_backend, 0);

  // INFERENCE
  VALUE cInference = rb_define_class_under(mLlamaste, "Inference", rb_cObject);
  rb_define_alloc_func(cInference, i_allocate);
  rb_define_method(cInference, "close", (VALUE(*)(ANYARGS))i_close, 0);
  rb_define_method(cInference, "process_tokens", (VALUE(*)(ANYARGS))i_process_tokens, 3);
}
