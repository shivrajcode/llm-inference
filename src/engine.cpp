#include "pocket/engine.hpp"
#include "pocket/policy.hpp"

#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

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

std::uint64_t available_memory() {
#if defined(__linux__)
    std::ifstream memory("/proc/meminfo");
    std::string key, unit;
    std::uint64_t value = 0;
    while (memory >> key >> value >> unit) {
        if (key == "MemAvailable:" && unit == "kB") return value * 1024;
    }
#elif defined(__APPLE__)
    const auto host = mach_host_self();
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_size = 0;
    const bool ok = host_page_size(host, &page_size) == KERN_SUCCESS &&
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) == KERN_SUCCESS;
    mach_port_deallocate(mach_task_self(), host);
    if (ok) {
        const auto available = (static_cast<std::uint64_t>(stats.free_count) + stats.inactive_count) * page_size;
        std::uint64_t physical = 0;
        std::size_t size = sizeof(physical);
        if (sysctlbyname("hw.memsize", &physical, &size, nullptr, 0) == 0)
            return std::min(available, physical);
        return available;
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) return memory.ullAvailPhys;
#endif
    return 0;
}

std::string metadata(const llama_model* model, const std::string& key) {
    std::array<char, 128> value{};
    const auto length = llama_model_meta_val_str(model, key.c_str(), value.data(), value.size());
    if (length <= 0 || static_cast<std::size_t>(length) >= value.size()) return {};
    return std::string(value.data(), static_cast<std::size_t>(length));
}

std::uint64_t model_kv_bytes(const llama_model* model) {
    const auto architecture = metadata(model, "general.architecture");
    const auto number = [&](const char* suffix) -> std::uint64_t {
        const auto text = metadata(model, architecture + suffix);
        std::uint64_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        return error == std::errc{} && end == text.data() + text.size() ? value : 0;
    };
    auto key = number(".attention.key_length");
    auto value = number(".attention.value_length");
    const auto embedding = number(".embedding_length");
    const auto heads = number(".attention.head_count");
    if (heads && embedding % heads == 0) {
        if (!key) key = embedding / heads;
        if (!value) value = embedding / heads;
    }
    return policy::kv_bytes_per_token(number(".block_count"), number(".attention.head_count_kv"), key, value);
}

// Remove one oldest completed turn without dropping system messages, the most
// recent user, or a partial user turn which has not yet received an answer.
int trim_oldest_turn(std::vector<llama_chat_message>& chat) {
    std::size_t latest = chat.size();
    for (std::size_t i = 0; i < chat.size(); ++i)
        if (std::string_view(chat[i].role) == "user") latest = i;
    for (std::size_t first = 0; first < latest; ++first) {
        if (std::string_view(chat[first].role) != "user") continue;
        std::size_t end = first + 1;
        bool answered = false;
        while (end < chat.size() && std::string_view(chat[end].role) != "user") {
            answered = answered || std::string_view(chat[end].role) == "assistant";
            ++end;
        }
        if (!answered) continue;
        int removed = 0;
        for (std::size_t i = end; i-- > first;) {
            if (std::string_view(chat[i].role) == "system") continue;
            chat.erase(chat.begin() + static_cast<std::ptrdiff_t>(i));
            ++removed;
        }
        return removed;
    }
    return 0;
}

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
    int effective_context = 0;
    int training_context = 0;
};

Engine::Engine(Config config) {
    if (config.context != 0 && (config.context < 256 || config.context > 32768))
        throw std::invalid_argument("Context must be 0 (automatic) or between 256 and 32768 tokens");
    if (config.threads < 1 || config.threads > 256)
        throw std::invalid_argument("Threads must be between 1 and 256");
    if (config.batch < 1 || (config.context != 0 && config.batch > config.context) || config.batch > 2048)
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
    impl_->effective_context = 0;
    impl_->training_context = 0;
}

bool Engine::loaded() const { return impl_->context != nullptr; }
std::string Engine::model_path() const { return impl_->path; }
Config Engine::config() const {
    auto effective = impl_->config;
    if (loaded()) {
        effective.context = impl_->effective_context;
        effective.batch = std::min(effective.batch, effective.context);
    }
    return effective;
}
int Engine::model_context() const { return impl_->training_context; }

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
    const auto available = available_memory();
    const auto model_bytes = std::filesystem::file_size(path);
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
    const auto training_context = llama_model_n_ctx_train(model.get());
    const auto chosen_context = impl_->config.context == 0
        ? policy::choose_context(training_context, available,
                                 std::max<std::uint64_t>(model_bytes, llama_model_size(model.get())), model_kv_bytes(model.get()))
        : impl_->config.context;
    if (chosen_context > training_context)
        throw std::invalid_argument("Requested context exceeds this model's training context; use automatic context");
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(chosen_context);
    ctx_params.n_batch = static_cast<uint32_t>(std::min(impl_->config.batch, chosen_context));
    ctx_params.n_ubatch = static_cast<uint32_t>(std::min({impl_->config.batch, chosen_context, 128}));
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = impl_->config.threads;
    ctx_params.n_threads_batch = impl_->config.threads;
    ctx_params.offload_kqv = impl_->config.gpu_layers > 0;
    Context context(llama_init_from_model(model.get(), ctx_params), llama_free);
    if (!context) throw std::runtime_error("Could not allocate inference context; reduce context or batch size");
    const auto actual_context = llama_n_ctx(context.get());
    if (actual_context > static_cast<std::uint32_t>(training_context))
        throw std::invalid_argument("Rounded context allocation exceeds model training context; use automatic context");
    impl_->model = std::move(model);
    impl_->context = std::move(context);
    impl_->path = std::move(path);
    impl_->effective_context = static_cast<int>(actual_context);
    impl_->training_context = training_context;
}

Result Engine::generate(const std::vector<Message>& messages, int max_tokens,
                        float temperature,
                        const std::function<bool(std::string_view)>& emit) {
    if (!loaded()) throw std::runtime_error("Load a model before starting a chat");
    if (!emit) throw std::invalid_argument("A generation callback is required");
    const auto effective = config();
    const bool auto_reply = max_tokens == 0;
    if (max_tokens < 0 || max_tokens > effective.context)
        throw std::invalid_argument("Output token limit must be 0 (automatic) or between 1 and context size");
    if (auto_reply) max_tokens = policy::reply_budget(effective.context);
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
    const auto* vocab = llama_model_get_vocab(impl_->model.get());
    std::string prompt;
    int32_t count = 0;
    int history_trimmed = 0;
    for (;;) {
        const auto needed = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
        if (needed <= 0 || needed > 2 * 1024 * 1024)
            throw std::invalid_argument("Unsupported chat template or formatted prompt too large");
        prompt.resize(static_cast<std::size_t>(needed));
        const auto written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, prompt.data(), needed);
        if (written != needed) throw std::runtime_error("Failed to format model chat template");
        const auto token_count = llama_tokenize(vocab, prompt.data(), needed, nullptr, 0, true, true);
        if (token_count >= 0 || token_count == INT32_MIN)
            throw std::runtime_error("Failed to tokenize chat prompt");
        count = -token_count;
        if (static_cast<std::uint64_t>(count) + max_tokens <= static_cast<std::uint64_t>(effective.context)) break;
        if (!auto_reply)
            throw std::invalid_argument("Chat history plus output limit exceeds context; use automatic output or start a new chat");
        const auto removed = trim_oldest_turn(chat);
        if (!removed && count < effective.context) {
            // Preserve a long latest message by reducing the reply allowance.
            max_tokens = effective.context - count;
            break;
        }
        if (!removed)
            throw std::invalid_argument("Your latest message and preserved instructions are too long for this model's available context; shorten the message or start a new chat");
        history_trimmed += removed;
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    if (llama_tokenize(vocab, prompt.data(), static_cast<int32_t>(prompt.size()), tokens.data(), count, true, true) != count)
        throw std::runtime_error("Failed to tokenize chat prompt");
    prompt.clear();
    prompt.shrink_to_fit();

    auto* context = impl_->context.get();
    llama_memory_clear(llama_get_memory(context), true);
    const ClearMemory cleanup{context};
    const auto start = std::chrono::steady_clock::now();
    Result result;
    result.prompt_tokens = count;
    result.context = effective.context;
    result.history_trimmed = history_trimmed;
    const auto finish = [&] {
        result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return result;
    };
    for (int32_t offset = 0; offset < count; offset += effective.batch) {
        if (!emit({})) return finish();
        auto batch = llama_batch_get_one(tokens.data() + offset,
                                        std::min(effective.batch, count - offset));
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
