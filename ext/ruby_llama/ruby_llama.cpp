#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#include "ggml.h"
#include "llama.h"
#include "ruby.h"
#include <ruby/thread.h>

struct model_params {
    int32_t  seed            = -1;     // seed
    int32_t  n_threads       = 4;      // threads
    int32_t  n_predict       = 128;    // tokens
    int32_t  repeat_last_n   = 64;     // penalize_tokens
    int32_t  n_parts         = 1;      // parts
    int32_t  n_ctx           = 128;    // context_size
    int32_t  top_k           = 40;     // top_k
    float    top_p           = 0.95f;  // top_p
    float    temp            = 0.80f;  // temperature
    float    repeat_penalty  = 1.10f;  // repeat_penalty
    int32_t  n_batch         = 8;      // batch_size
    bool     use_mlock       = false;  // memory_lock
    bool     memory_f16      = false;  // memory_f16
    bool     use_mmap        = true;
    bool     embedding       = false;
    char*    lora_base       = "";
    char*    lora_adapter    = "";
    char*    model_path;               // model
};

struct lm_typedata {
    llama_context *ctx;
    model_params  *params;
};

struct process_token_data {
    std::vector<llama_token> tokens;
    std::vector<std::string> break_on;
    lm_typedata *typedata;
    bool locked;
    bool ingest;
    VALUE block;
    std::string output_str;
};

static const rb_data_type_t lm_type = {
    "LlamaData",
    {0, [](void *p) { delete (lm_typedata *)p; }, 0},
    0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

static std::vector<llama_token> llama_tokenize(struct llama_context * ctx, const std::string & text, bool add_bos) {
    std::vector<llama_token> res(text.size() + (int)add_bos);
    int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
    assert(n >= 0);
    res.resize(n);

    return res;
}

static VALUE embeds_to_rb_array(const float* embeds, int embd_size) {
    VALUE ruby_array = rb_ary_new2(embd_size);

    for(int i = 0; i < embd_size; ++i) { rb_ary_push(ruby_array, DBL2NUM(embeds[i])); }

    return ruby_array;
}

static VALUE tokens_to_rb_array(struct llama_context * ctx, const std::vector<llama_token>& tokens) {
    VALUE ruby_array = rb_ary_new2(tokens.size());

    for(const auto& token : tokens) {
        VALUE tuple = rb_ary_new2(2);

        rb_ary_push(tuple, rb_str_new_cstr(llama_token_to_str(ctx, token)));
        rb_ary_push(tuple, INT2NUM(token));
        rb_ary_push(ruby_array, tuple);
    }

    return ruby_array;
}

static std::vector<llama_token> rb_array_to_vector(VALUE ruby_array) {
    std::vector<llama_token> tokens;
    long len = RARRAY_LEN(ruby_array);

    for(long i = 0; i < len; ++i) {
        VALUE element = rb_ary_entry(ruby_array, i);
        tokens.push_back(NUM2INT(element));
    }

    return tokens;
}

static std::vector<std::string> rb_array_of_break_strings(VALUE break_on_str) {
    std::vector<std::string> cpp_strings;

    if(break_on_str != Qnil) {
        for(int i = 0; i < RARRAY_LEN(break_on_str); i++) {
            VALUE ruby_str = rb_ary_entry(break_on_str, i);
            char* c_str = StringValueCStr(ruby_str);
            cpp_strings.push_back(c_str);
        }
    }

    return cpp_strings;
}

static void *resync_execute_block(void *data) {
    VALUE *str = (VALUE *)data;
    rb_yield(*str);
}

std::vector<llama_token> consume_input(std::vector<llama_token>& embd_inp, int& input_consumed, model_params* params) {
    std::vector<llama_token> embd;

    while(embd_inp.size() > input_consumed) {
        embd.push_back(embd_inp[input_consumed]);

        ++input_consumed;

        if(embd.size() >= params->n_batch) { break; }
    }

    return embd;
}

llama_token generate_token(llama_context* ctx, std::vector<llama_token>& last_n_tokens, int n_ctx, model_params* params) {
    return llama_sample_top_p_top_k(
        ctx,
        last_n_tokens.data() + n_ctx - params->repeat_last_n,
        params->repeat_last_n,
        params->top_k,
        params->top_p,
        params->temp,
        params->repeat_penalty
    );
}

bool check_break_conditions(const std::string& token, std::vector<std::string>& break_on, int tokens_generated, llama_token id) {
    if(id == llama_token_eos() && tokens_generated > 1) {
        return true;
    }

    for (std::string& brk : break_on) {
        if(token.find(brk) != std::string::npos) {
            return true;
        }
    }

    return false;
}

static std::string process_tokens(std::vector<llama_token> embd_inp, lm_typedata* typedata, bool locked, bool ingest, VALUE callback, std::vector<std::string> break_on) {
    std::ostringstream output_buffer;

    llama_context *ctx = typedata->ctx;
    model_params *params = typedata->params;

    if(params->n_threads == 0){ return ""; }

    const int n_ctx = llama_n_ctx(ctx);
    params->n_predict = std::min(params->n_predict, n_ctx - (int) embd_inp.size());
    int remaining_tokens = params->n_predict;

    std::vector<llama_token> embd;
    bool break_early = false;

    int last_n_size = params->repeat_last_n;
    std::vector<llama_token> last_n_tokens(last_n_size);

    if(ingest) { std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0); }
    else { last_n_tokens = embd_inp; }

    int tokens_generated = 0;
    int n_past = (ingest) ? 0 : embd_inp.size();
    int input_consumed = (ingest) ? 0 : embd_inp.size();

    while (remaining_tokens > tokens_generated) {
        if (embd.size() > 0) {
            if (llama_eval(ctx, embd.data(), embd.size(), n_past, params->n_threads)) {
                break;
            }
        }

        n_past += embd.size();
        embd.clear();

        if(embd_inp.size() > input_consumed) { embd = consume_input(embd_inp, input_consumed, params); }
        else {
            llama_token id = generate_token(ctx, last_n_tokens, n_ctx, params);

            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(id);

            embd.push_back(id);
            tokens_generated++;

            std::string token = llama_token_to_str(ctx, id);
            output_buffer << token;

            if(callback != Qnil) {
                VALUE rb_str = rb_str_new_cstr(token.c_str());

                if(locked) { rb_thread_call_with_gvl(resync_execute_block, (void *)&rb_str); }
                else { rb_yield(rb_str); }
            }

            if(check_break_conditions(token, break_on, tokens_generated, id)) { break; }
        }
    }

    std::string output_string = output_buffer.str();
    return output_string;
}

static int hash_iter_callback(VALUE key, VALUE value, VALUE params_ptr) {
    model_params* params = (model_params*)params_ptr;

    std::string key_str = rb_id2name(SYM2ID(key));

    if (key_str == "model") { params->model_path = StringValueCStr(value); }
    else if (key_str == "seed") { params->seed = NUM2INT(value); }
    else if (key_str == "threads") { params->n_threads = NUM2INT(value); }
    else if (key_str == "tokens") { params->n_predict = NUM2INT(value); }
    else if (key_str == "penalize_tokens") { params->repeat_last_n = NUM2INT(value); }
    else if (key_str == "parts") { params->n_parts = NUM2INT(value); }
    else if (key_str == "context_size") { params->n_ctx = NUM2INT(value); }
    else if (key_str == "top_k") { params->top_k = (float) NUM2INT(value); }
    else if (key_str == "top_p") { params->top_p = (float) RFLOAT_VALUE(value); }
    else if (key_str == "temperature") { params->temp = (float) RFLOAT_VALUE(value); }
    else if (key_str == "repeat_penalty") { params->repeat_penalty = (float) RFLOAT_VALUE(value); }
    else if (key_str == "batch_size") { params->n_batch = NUM2INT(value); }
    else if (key_str == "memory_lock") { params->use_mlock = RTEST(value); }
    else if (key_str == "memory_f16") { params->memory_f16 = RTEST(value); }
    else if (key_str == "embedding") { params->embedding = RTEST(value); }
    else if (key_str == "lora_base") { params->lora_base = (value == Qnil) ? params->lora_base : StringValueCStr(value); }
    else if (key_str == "lora_adapter") { params->lora_adapter = (value == Qnil) ? params->lora_adapter : StringValueCStr(value); }

    return ST_CONTINUE;
}

void ruby_params_parse(VALUE ruby_hash, model_params &params) {
    rb_hash_foreach(ruby_hash, [](VALUE key, VALUE value, VALUE params_ptr) -> int {
            return hash_iter_callback(key, value, params_ptr);
            }, (VALUE)&params);
}

static void *process_tokens_async(void *data) {
    process_token_data *pt_data = (process_token_data *)data;

    pt_data->output_str = process_tokens(
            pt_data->tokens,
            pt_data->typedata,
            pt_data->locked,
            pt_data->ingest,
            pt_data->block,
            pt_data->break_on
            );

    return NULL;
}

static VALUE m_initialize(VALUE self, VALUE params_hash) {
    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    auto params = std::make_unique<model_params>();
    ruby_params_parse(params_hash, *params);

    typedata->params = params.release();

    return self;
}

static VALUE m_close(VALUE self) {
    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    llama_free(typedata->ctx);
    delete(typedata->params);
    ruby_xfree(typedata);

    return Qnil;
}

static VALUE m_allocate(VALUE klass) {
    lm_typedata *typedata = static_cast<lm_typedata*>(ruby_xmalloc(sizeof(lm_typedata)));

    return TypedData_Make_Struct(klass, lm_typedata, &lm_type, typedata);
}

static VALUE m_quantize(VALUE self, VALUE input_file, VALUE output_file, VALUE quant_type) {
    char* input_fp = StringValueCStr(input_file);
    char* output_fp = StringValueCStr(output_file);
    llama_ftype itype = (llama_ftype) NUM2INT(quant_type);

    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    if(llama_model_quantize(input_fp, output_fp, itype, typedata->params->n_threads)) { return Qnil; }
    else { return output_file; }
}

static VALUE m_load_model(VALUE self) {
    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    auto lparams = llama_context_default_params();

    lparams.n_ctx      = typedata->params->n_ctx;
    lparams.n_parts    = typedata->params->n_parts;
    lparams.seed       = typedata->params->seed;
    lparams.f16_kv     = typedata->params->memory_f16;
    lparams.use_mlock  = typedata->params->use_mlock;
    lparams.use_mlock  = typedata->params->use_mlock;
    lparams.use_mmap   = typedata->params->use_mmap;
    lparams.embedding  = typedata->params->embedding;
    lparams.logits_all = false;

    typedata->ctx = llama_init_from_file(
            typedata->params->model_path,
            lparams
    );

    if (!strlen(typedata->params->lora_adapter) == 0) {
        char * l_base = (strlen(typedata->params->lora_base) == 0) ? NULL : typedata->params->lora_base;

        llama_apply_lora_from_file(
                typedata->ctx,
                typedata->params->lora_adapter,
                typedata->params->lora_base,
                typedata->params->n_threads
        );
    }

    return self;
}

static VALUE m_embedding(VALUE self, VALUE input_text) {
    Check_Type(input_text, T_STRING);

    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);
    auto tokens_embd = ::llama_tokenize(typedata->ctx, StringValueCStr(input_text), true);

    llama_eval(
        typedata->ctx,
        tokens_embd.data(),
        tokens_embd.size(),
        0,
        typedata->params->n_threads
    );

    const int n_embd = llama_n_embd(typedata->ctx);
    const float* embeddings = llama_get_embeddings(typedata->ctx);

    return embeds_to_rb_array(embeddings, n_embd);
}

static VALUE m_tokenize(VALUE self, VALUE input_text) {
    Check_Type(input_text, T_STRING);

    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    auto tokens_embd = ::llama_tokenize(typedata->ctx, StringValueCStr(input_text), false);

    return tokens_to_rb_array(typedata->ctx, tokens_embd);
}

static VALUE m_cache_prompt(VALUE self, VALUE input_prompt) {
    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    auto tokens = rb_array_to_vector(input_prompt);

    llama_eval(
        typedata->ctx,
        tokens.data(),
        tokens.size(),
        0,
        typedata->params->n_threads
    );

    size_t state_size = llama_get_state_size(typedata->ctx);
    std::vector<uint8_t> state_data(state_size);

    size_t bytes_copied = llama_copy_state_data(typedata->ctx, state_data.data());

    return rb_str_new(reinterpret_cast<const char*>(state_data.data()), state_size);
}

static VALUE m_resume_prompt(VALUE self, VALUE input_prompt, VALUE rb_state_data, VALUE break_on_str) {
    Check_Type(rb_state_data, T_STRING);

    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    long state_data_length = RSTRING_LEN(rb_state_data);
    uint8_t *state_data = new uint8_t[state_data_length];
    memcpy(state_data, RSTRING_PTR(rb_state_data), state_data_length);

    size_t bytes_read = llama_set_state_data(typedata->ctx, state_data);

    process_token_data pt_data;
    pt_data.tokens = rb_array_to_vector(input_prompt);
    pt_data.break_on = rb_array_of_break_strings(break_on_str);
    pt_data.typedata = typedata;
    pt_data.locked = true;
    pt_data.ingest = false;
    pt_data.block = rb_block_given_p() ? rb_block_proc() : Qnil;

    rb_thread_call_without_gvl(
        process_tokens_async,
        (void *)&pt_data,
        RUBY_UBF_IO,
        0
    );

    return rb_str_new_cstr(pt_data.output_str.c_str());
}

static VALUE m_process_tokens(VALUE self, VALUE input_tokens, VALUE break_on_str) {
    lm_typedata *typedata;
    TypedData_Get_Struct(self, lm_typedata, &lm_type, typedata);

    process_token_data pt_data;
    pt_data.tokens = rb_array_to_vector(input_tokens);
    pt_data.break_on = rb_array_of_break_strings(break_on_str);
    pt_data.typedata = typedata;
    pt_data.locked = true;
    pt_data.ingest = true;
    pt_data.block = rb_block_given_p() ? rb_block_proc() : Qnil;

    rb_thread_call_without_gvl(
            process_tokens_async,
            (void *)&pt_data,
            RUBY_UBF_IO,
            0
    );

    return rb_str_new_cstr(pt_data.output_str.c_str());
}

extern "C" void Init_ruby_llama() {
    VALUE cLlama = rb_define_class("Llama", rb_cObject);

    rb_define_alloc_func(cLlama, m_allocate);
    rb_define_method(cLlama, "initialize", (VALUE(*)(ANYARGS))m_initialize, 1);
    rb_define_method(cLlama, "load_model", (VALUE(*)(ANYARGS))m_load_model, 0);
    rb_define_method(cLlama, "process_tokens", (VALUE(*)(ANYARGS))m_process_tokens, 2);
    rb_define_method(cLlama, "cache_prompt", (VALUE(*)(ANYARGS))m_cache_prompt, 1);
    rb_define_method(cLlama, "resume_prompt", (VALUE(*)(ANYARGS))m_resume_prompt, 3);
    rb_define_method(cLlama, "tokenize_text", (VALUE(*)(ANYARGS))m_tokenize, 1);
    rb_define_method(cLlama, "embed_text", (VALUE(*)(ANYARGS))m_embedding, 1);
    rb_define_method(cLlama, "quantize", (VALUE(*)(ANYARGS))m_quantize, 3);
    rb_define_method(cLlama, "close", (VALUE(*)(ANYARGS))m_close, 0);
}
