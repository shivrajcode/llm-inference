#include "pocket/engine.hpp"

#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>

namespace pocket {
namespace {
struct Backend {
    Backend() { llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

std::shared_ptr<Backend> acquire_backend() {
    static std::mutex mutex;
    static std::weak_ptr<Backend> weak;
    const std::lock_guard lock(mutex);
    auto backend = weak.lock();
    if (!backend) {
        backend = std::make_shared<Backend>();
        weak = backend;
    }
    return backend;
}

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

// Tokens may split UTF-8 characters. Retain incomplete suffixes between emits
// and replace malformed sequences rather than handing invalid text to JSON.
std::string take_utf8(std::string& pending, bool final = false) {
    std::string output;
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto first = static_cast<unsigned char>(pending[offset]);
        std::size_t count = first < 0x80 ? 1 : first >= 0xC2 && first <= 0xDF ? 2
            : first >= 0xE0 && first <= 0xEF ? 3 : first >= 0xF0 && first <= 0xF4 ? 4 : 0;
        if (count && pending.size() - offset < count && !final) break;
        bool valid = count != 0 && pending.size() - offset >= count;
        for (std::size_t i = 1; valid && i < count; ++i) {
            const auto byte = static_cast<unsigned char>(pending[offset + i]);
            valid = (byte & 0xC0) == 0x80;
        }
        if (valid && count >= 3) {
            const auto second = static_cast<unsigned char>(pending[offset + 1]);
            valid = !(first == 0xE0 && second < 0xA0) && !(first == 0xED && second >= 0xA0)
                && !(first == 0xF0 && second < 0x90) && !(first == 0xF4 && second >= 0x90);
        }
        if (valid) { output.append(pending, offset, count); offset += count; }
        else { output += "\xEF\xBF\xBD"; ++offset; }
    }
    pending.erase(0, offset);
    return output;
}

struct ClearMemory {
    llama_context* context;
    ~ClearMemory() { llama_memory_clear(llama_get_memory(context), true); }
};
} // namespace

struct Engine::Impl {
    explicit Impl(Config value) : config(value), backend(acquire_backend()) {}
    Config config;
    std::shared_ptr<Backend> backend;
    Model model{nullptr, llama_model_free};
    Context context{nullptr, llama_free};
    std::string path;
};

Engine::Engine(Config config) {
    if (config.context < 256 || config.context > 32768)
        throw std::invalid_argument("Context must be between 256 and 32768 tokens");
    if (config.threads < 1 || config.threads > 256)
        throw std::invalid_argument("Threads must be between 1 and 256");
    if (config.batch < 1 || config.batch > config.context || config.batch > 2048)
        throw std::invalid_argument("Batch must be between 1 and min(context, 2048)");
    if (config.gpu_layers < 0 || config.gpu_layers > 1024)
        throw std::invalid_argument("GPU layers must be between 0 and 1024");
    impl_ = std::make_unique<Impl>(config);
}

Engine::~Engine() = default;

void Engine::unload() {
    impl_->context.reset();
    impl_->model.reset();
    impl_->path.clear();
}

bool Engine::loaded() const { return impl_->context != nullptr; }
std::string Engine::model_path() const { return impl_->path; }

void Engine::load(std::string path) {
    if (path.empty() || path.find('\0') != std::string::npos)
        throw std::invalid_argument("Model path is empty or invalid");
    if (!std::filesystem::is_regular_file(path))
        throw std::invalid_argument("Model path must be an existing regular GGUF file");
    std::ifstream file(path, std::ios::binary);
    std::array<char, 4> magic{};
    if (!file.read(magic.data(), magic.size()) || std::string_view(magic.data(), 4) != "GGUF")
        throw std::invalid_argument("Model file is not GGUF");
    file.close();

    // Avoid temporarily retaining two models on devices with limited RAM.
    unload();
    auto params = llama_model_default_params();
    params.n_gpu_layers = impl_->config.gpu_layers;
    params.use_mmap = true;
    params.use_mlock = false;
    Model model(llama_model_load_from_file(path.c_str(), params), llama_model_free);
    if (!model) throw std::runtime_error("Could not load GGUF model; check format, compatibility and available memory");
    if (llama_model_has_encoder(model.get()) || !llama_model_has_decoder(model.get()))
        throw std::invalid_argument("Only decoder-only text generation models are supported");
    if (!llama_model_chat_template(model.get(), nullptr))
        throw std::invalid_argument("Model has no chat template; use an instruction/chat GGUF model");
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(impl_->config.context);
    ctx_params.n_batch = static_cast<uint32_t>(impl_->config.batch);
    ctx_params.n_ubatch = static_cast<uint32_t>(std::min(impl_->config.batch, 128));
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = impl_->config.threads;
    ctx_params.n_threads_batch = impl_->config.threads;
    ctx_params.offload_kqv = impl_->config.gpu_layers > 0;
    Context context(llama_init_from_model(model.get(), ctx_params), llama_free);
    if (!context) throw std::runtime_error("Could not allocate inference context; reduce context or batch size");
    impl_->model = std::move(model);
    impl_->context = std::move(context);
    impl_->path = std::move(path);
}

Result Engine::generate(const std::vector<Message>& messages, int max_tokens,
                        float temperature,
                        const std::function<bool(std::string_view)>& emit) {
    if (!loaded()) throw std::runtime_error("Load a model before starting a chat");
    if (!emit) throw std::invalid_argument("A generation callback is required");
    if (max_tokens < 1 || max_tokens > impl_->config.context)
        throw std::invalid_argument("Output token limit must be between 1 and context size");
    if (!std::isfinite(temperature) || temperature < 0 || temperature > 2)
        throw std::invalid_argument("Temperature must be between 0 and 2");
    if (messages.empty() || messages.size() > 256)
        throw std::invalid_argument("Supply between 1 and 256 chat messages");
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    std::size_t total_bytes = 0;
    for (const auto& message : messages) {
        if (message.role != "system" && message.role != "user" && message.role != "assistant")
            throw std::invalid_argument("Chat roles must be system, user or assistant");
        if (message.content.find('\0') != std::string::npos)
            throw std::invalid_argument("Chat messages cannot contain NUL bytes");
        if (message.content.size() > 1024 * 1024 - total_bytes)
            throw std::invalid_argument("Chat history exceeds the 1 MiB input limit");
        total_bytes += message.content.size();
        chat.push_back({message.role.c_str(), message.content.c_str()});
    }
    const auto* tmpl = llama_model_chat_template(impl_->model.get(), nullptr);
    int32_t needed = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
    if (needed <= 0 || needed > 2 * 1024 * 1024)
        throw std::invalid_argument("Unsupported chat template or formatted prompt too large");
    std::string prompt(static_cast<std::size_t>(needed), '\0');
    const auto written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                                  prompt.data(), needed);
    if (written != needed) throw std::runtime_error("Failed to format model chat template");
    const auto* vocab = llama_model_get_vocab(impl_->model.get());
    const int32_t token_count = llama_tokenize(vocab, prompt.data(), needed, nullptr, 0, true, true);
    if (token_count >= 0 || token_count == INT32_MIN)
        throw std::runtime_error("Failed to tokenize chat prompt");
    const int32_t count = -token_count;
    if (static_cast<uint64_t>(count) + static_cast<uint64_t>(max_tokens) > llama_n_ctx(impl_->context.get()))
        throw std::invalid_argument("Chat history plus output limit exceeds context; start a new chat or reduce output tokens");
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    if (llama_tokenize(vocab, prompt.data(), needed, tokens.data(), count, true, true) != count)
        throw std::runtime_error("Failed to tokenize chat prompt");
    prompt.clear();
    prompt.shrink_to_fit();

    auto* context = impl_->context.get();
    llama_memory_clear(llama_get_memory(context), true);
    const ClearMemory cleanup{context};
    const auto start = std::chrono::steady_clock::now();
    Result result;
    result.prompt_tokens = count;
    const auto finish = [&] {
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    };
    for (int32_t offset = 0; offset < count; offset += impl_->config.batch) {
        if (!emit({})) return finish();
        auto batch = llama_batch_get_one(tokens.data() + offset,
                                        std::min(impl_->config.batch, count - offset));
        if (llama_decode(context, batch) != 0)
            throw std::runtime_error("Prompt evaluation failed; try a smaller context or batch");
    }
    Sampler sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (!sampler) throw std::runtime_error("Could not allocate token sampler");
    if (temperature == 0) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    std::string pending;
    std::vector<char> piece(256);
    for (int i = 0; i < max_tokens; ++i) {
        if (!emit({})) return finish();
        auto token = llama_sampler_sample(sampler.get(), context, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        ++result.generated_tokens;
        int32_t size = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        if (size < 0) {
            if (size == INT32_MIN || -size > 1024 * 1024)
                throw std::runtime_error("Model produced an oversized token");
            piece.resize(static_cast<std::size_t>(-size));
            size = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
        }
        if (size < 0) throw std::runtime_error("Could not decode model token");
        pending.append(piece.data(), static_cast<std::size_t>(size));
        const auto text = take_utf8(pending);
        if (!text.empty() && !emit(text)) return finish();
        if (i + 1 < max_tokens && llama_decode(context, llama_batch_get_one(&token, 1)) != 0)
            throw std::runtime_error("Token evaluation failed");
    }
    const auto tail = take_utf8(pending, true);
    if (!tail.empty()) emit(tail);
    return finish();
}

} // namespace pocket
