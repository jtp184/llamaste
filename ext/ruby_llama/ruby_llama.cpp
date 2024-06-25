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
  std::string model_path;
  llama_model *model;
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

static VALUE m_load_model(VALUE self) {
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  fprintf(stderr, "Loading model params\n");
  llama_model_params m_params = llama_model_default_params();

  VALUE model_path = rb_iv_get(self, "@model_path");
  // THIS IS WHERE IT FAILS
  data->model_path = StringValueCStr(model_path);
  fprintf(stderr, "Model path: %s\n", StringValueCStr(model_path));
  VALUE use_mmap = rb_iv_get(self, "@use_mmap");
  m_params.use_mmap = RTEST(use_mmap);
  fprintf(stderr, "Use mmap: %d\n", RTEST(use_mmap));
  VALUE use_mlock = rb_iv_get(self, "@use_mlock");
  m_params.use_mlock = RTEST(use_mlock);
  fprintf(stderr, "Use mlock: %d\n", RTEST(use_mlock));

  fprintf(stderr, "Loading model\n");
  data->model = llama_load_model_from_file(
    data->model_path.c_str(),
    m_params
  );

  if (data->model == NULL) { return Qfalse; }

  return Qtrue;
}

static VALUE m_allocate(VALUE klass) {
  model_data *data = (model_data*)ruby_xmalloc(sizeof(model_data));

  data->model = NULL;

  return TypedData_Wrap_Struct(klass, &model_type, data);
}

static VALUE m_close(VALUE self) {
  /* fprintf(stderr, "Closing model\n"); */
  model_data *data;
  TypedData_Get_Struct(self, model_data, &model_type, data);

  llama_free_model(data->model);
  data->model = NULL;
  ruby_xfree(data);

  return Qnil;
}

// INFERENCE

// Model and shared context
struct inference_data {
  model_data *m_data;
  llama_context_params c_params;
  llama_context *ctx;
};

// Per-run context
struct inference_context {
  inference_data *i_data;
  std::string prompt;
  int n_predict;
  bool callback_block;
  llama_batch batch;
  std::vector<llama_token> tokens;
  std::vector<llama_token> output_tokens;
  std::string output_str;
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

static int i_parse_params_iter_callback(VALUE key, VALUE value, VALUE self) {
  inference_data *data;
  TypedData_Get_Struct(self, inference_data, &inference_type, data);

  std::string key_str = rb_id2name(SYM2ID(key));

  // TODO: More params from context
  if(key_str == "seed") { data->c_params.seed = NUM2INT(value); }
  else if(key_str == "ctx" || key_str == "n_ctx" || key_str == "context") { data->c_params.n_ctx = NUM2INT(value); }
  else if(key_str == "batch" || key_str == "n_batch") { data->c_params.n_batch = NUM2INT(value); }
  else if(key_str == "ubatch" || key_str == "n_ubatch") { data->c_params.n_ubatch = NUM2INT(value); }
  else if(key_str == "seq_max" || key_str == "n_seq_max") { data->c_params.n_seq_max = NUM2INT(value); }
  else if(key_str == "threads" || key_str == "n_threads") { data->c_params.n_threads = NUM2INT(value); }
  else if(key_str == "threads_batch" || key_str == "n_threads_batch") { data->c_params.n_threads_batch = NUM2INT(value); }
       //*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
       //*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
  else if(key_str == "rope_freq_base") { data->c_params.rope_freq_base = NUM2DBL(value); }
  else if(key_str == "rope_freq_scale") { data->c_params.rope_freq_scale = NUM2DBL(value); }
  else if(key_str == "yarn_ext_factor") { data->c_params.yarn_ext_factor = NUM2DBL(value); }
  else if(key_str == "yarn_attn_factor") { data->c_params.yarn_attn_factor = NUM2DBL(value); }
  else if(key_str == "yarn_beta_fast") { data->c_params.yarn_beta_fast = NUM2DBL(value); }
  else if(key_str == "yarn_beta_slow") { data->c_params.yarn_beta_slow = NUM2DBL(value); }
  else if(key_str == "yarn_orig_ctx") { data->c_params.yarn_orig_ctx = NUM2INT(value); }
  else if(key_str == "defrag_thold") { data->c_params.defrag_thold = NUM2DBL(value); }
  else if(key_str == "flash_attn") { data->c_params.flash_attn = RTEST(value); }
  else if(key_str == "embeddings") { data->c_params.embeddings = RTEST(value); }

  return ST_CONTINUE;
}

static void i_apply_params(VALUE self, VALUE params_hash) {
  rb_hash_foreach(
    params_hash,
    i_parse_params_iter_callback,
    self
  );
}

static int i_get_predict(VALUE self) {
  VALUE options = rb_iv_get(self, "@options");

  VALUE predict_key = Qnil;
  const char* allowed_keys[] = {"predict", "n_predict", "tokens"};
  int num_keys = sizeof(allowed_keys) / sizeof(allowed_keys[0]);

  for (int i = 0; i < num_keys; i++) {
    VALUE key = ID2SYM(rb_intern(allowed_keys[i]));
    if (!NIL_P(rb_hash_aref(options, key))) {
      predict_key = key;
      break;
    }
  }

  if (NIL_P(predict_key)) {
    rb_raise(rb_eKeyError, "Must supply one of 'predict', 'n_predict', or 'tokens' in @options");
  }

  VALUE predict_value = rb_hash_aref(options, predict_key);

  return NUM2INT(predict_value);
}

static void i_init_inference_shared(VALUE self) {
  inference_data *data;
  TypedData_Get_Struct(self, inference_data, &inference_type, data);

  if(data->m_data == NULL) {
    VALUE model = rb_iv_get(self, "@model");
    model_data *m_data;
    TypedData_Get_Struct(model, model_data, &model_type, m_data);

    data->m_data = m_data;
  }

  if(data->ctx == NULL) {
    VALUE options = rb_iv_get(self, "@options");
    if (!RB_TYPE_P(options, T_HASH)) { rb_raise(rb_eTypeError, "@options must be a Hash"); }

    i_apply_params(self, options);

    data->ctx = llama_new_context_with_model(
      data->m_data->model,
      data->c_params
    );
  }
}

static inference_context i_init_inference_context() {
  inference_context i_context;
  new (&i_context) inference_context();

  i_context.n_predict = 0;
  i_context.callback_block = false;
  new (&i_context.prompt) std::string();
  new (&i_context.tokens) std::vector<llama_token>();
  new (&i_context.output_str) std::string();
  new (&i_context.output_tokens) std::vector<llama_token>();
  new (&i_context.batch) llama_batch();

  return i_context;
}

static std::string i_predict_tokens(inference_context *i_context) {
  // Buffer for output
  std::ostringstream output_buffer;

  // Create initial tokens
  i_context->tokens = ::llama_tokenize(
      i_context->i_data->ctx,
      i_context->prompt.c_str(),
      true
  );

  // TODO: Does this batch size matter?
  i_context->batch = llama_batch_init(512, 0, 1);

  for(size_t i = 0; i < i_context->tokens.size(); i++) {
    llama_batch_add(i_context->batch, i_context->tokens[i], i, { 0 }, false);
  }

  // llama_decode will output logits only for the last token of the prompt
  // TODO: Play with this
  i_context->batch.logits[i_context->batch.n_tokens - 1] = true;

  // Decode initial prompt
  llama_decode(i_context->i_data->ctx, i_context->batch);

  // Counters
  int n_cur    = i_context->batch.n_tokens;
  int n_decode = 0;

  // Main loop
  while (n_cur <= i_context->n_predict) {
    {
      auto n_vocab = llama_n_vocab(i_context->i_data->m_data->model);
      auto *logits = llama_get_logits_ith(i_context->i_data->ctx, i_context->batch.n_tokens - 1);

      std::vector<llama_token_data> candidates;
      candidates.reserve(n_vocab);

      for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
      }

      llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

      // sample the most likely token
      const llama_token new_token_id = llama_sample_token_greedy(i_context->i_data->ctx, &candidates_p);

      // Add token to outputs
      std::string token = llama_token_to_piece(i_context->i_data->ctx, new_token_id);
      output_buffer << token;
      i_context->output_tokens.push_back(new_token_id);

      // Call callback block if it exists
      if(i_context->callback_block) {
        VALUE rb_str = rb_str_new_cstr(token.c_str());

        rb_thread_call_with_gvl(
          i_resync_execute_block,
          (void *)&rb_str
        );
      }

      // is it an end of generation?
      if (llama_token_is_eog(i_context->i_data->m_data->model, new_token_id) || n_cur == i_context->n_predict) {
        break;
      }

      // prepare the next batch
      llama_batch_clear(i_context->batch);

      // add next token to eval
      llama_batch_add(i_context->batch, new_token_id, n_cur, { 0 }, true);

      n_decode += 1;
    }

    n_cur += 1;

    // Evaluate current batch with transformer model
    llama_decode(i_context->i_data->ctx, i_context->batch);
  }

  // Return final output
  std::string output = output_buffer.str();
  return output;
}

static void *i_async_process_tokens(void *data) {
  inference_context *i_context = (inference_context *)data;

  i_context->output_str = i_predict_tokens(i_context);

  return NULL;
}

static VALUE i_process_tokens(VALUE self) {
  inference_data *i_data;
  TypedData_Get_Struct(self, inference_data, &inference_type, i_data);

  inference_context i_context;
  new (&i_context) inference_context();

  i_init_inference_shared(self);

  i_context.i_data = i_data;
  VALUE prompt = rb_iv_get(self, "@prompt");
  i_context.prompt = StringValueCStr(prompt);
  i_context.callback_block = rb_block_given_p();
  i_context.n_predict = i_get_predict(self);

  rb_thread_call_without_gvl(
    i_async_process_tokens,
    (void *)&i_context,
    RUBY_UBF_IO,
    0
  );

  // Free
  i_context.prompt.~basic_string();
  i_context.tokens.~vector();
  i_context.output_str.~basic_string();
  i_context.output_tokens.~vector();

  llama_batch_free(i_context.batch);

  // Return string and tokens
  VALUE tuple = rb_ary_new2(2);
  rb_ary_push(tuple, rb_str_new_cstr(i_context.output_str.c_str()));
  rb_ary_push(tuple, tokens_to_rb_array(i_data->ctx, i_context.output_tokens));

  return tuple;
}

static VALUE i_tokenize_text(VALUE self, VALUE input_str) {
  inference_data *data;
  TypedData_Get_Struct(self, inference_data, &inference_type, data);

  i_init_inference_shared(self);

  auto tokens_list = ::llama_tokenize(data->ctx, StringValueCStr(input_str), true);

  return tokens_to_rb_array(data->ctx, tokens_list);
}

static VALUE i_close(VALUE self) {
  /* fprintf(stderr, "Closing inference\n"); */
  inference_data *data ;
  TypedData_Get_Struct(self, inference_data, &inference_type, data);

  llama_free(data->ctx);
  data->ctx = NULL;
  ruby_xfree(data);

  return Qnil;
}

static VALUE i_allocate(VALUE klass) {
  inference_data *data = static_cast<inference_data*>(ruby_xmalloc(sizeof(inference_data)));

  data->m_data = NULL;
  data->ctx = NULL;
  data->c_params = llama_context_default_params();

  return TypedData_Wrap_Struct(klass, &inference_type, data);
}

// DEFINITIONS

extern "C" void Init_ruby_llama() {
  VALUE mLlamaste = rb_define_module("Llamaste");

  // MODEL
  VALUE cModel = rb_define_class_under(mLlamaste, "Model", rb_cObject);
  rb_define_alloc_func(cModel, m_allocate);
  rb_define_method(cModel, "close", (VALUE(*)(ANYARGS))m_close, 0);
  rb_define_method(cModel, "load_model", (VALUE(*)(ANYARGS))m_load_model, 0);
  rb_define_singleton_method(cModel, "start_backend", m_start_backend, -1);
  rb_define_singleton_method(cModel, "stop_backend", m_stop_backend, 0);

  // INFERENCE
  VALUE cInference = rb_define_class_under(mLlamaste, "Inference", rb_cObject);
  rb_define_alloc_func(cInference, i_allocate);
  rb_define_method(cInference, "close", (VALUE(*)(ANYARGS))i_close, 0);
  rb_define_method(cInference, "tokenize_text", (VALUE(*)(ANYARGS))i_tokenize_text, 1);
  rb_define_method(cInference, "process_tokens", (VALUE(*)(ANYARGS))i_process_tokens, 0);
}
