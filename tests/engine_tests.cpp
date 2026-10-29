#include "pocket/engine.hpp"
#include "pocket/policy.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <limits>

namespace {
void require(bool value, const char* description) {
    if (!value) throw std::runtime_error(description);
}

template<class F> void rejects(F&& operation, const char* description) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    require(rejected, description);
}
}

int main(int argc, char** argv) {
    try {
        using namespace pocket::policy;
        require(kv_bytes_per_token(28, 8, 128, 128) == 114688, "incorrect F16 KV estimate");
        require(kv_bytes_per_token(0, 8, 128, 128) == fallback_kv_bytes_per_token, "missing metadata fallback failed");
        require(kv_bytes_per_token(100, 100, std::numeric_limits<std::uint64_t>::max(), 1) ==
                std::numeric_limits<std::uint64_t>::max(), "KV arithmetic overflowed");
        require(choose_context(32768, 4096 * mib, 512 * mib, 65536) == 8192, "automatic upper cap failed");
        require(choose_context(3000, 4096 * mib, 512 * mib, 65536) == 2816, "model context rounding failed");
        require(choose_context(32768, 1024 * mib, 256 * mib, 262144) == 512, "tight RAM policy failed");
        require(choose_context(32768, 0, 256 * mib, 262144) == 512, "unknown memory fallback failed");
        rejects([] { choose_context(32768, 1024 * mib, 768 * mib, 65536); }, "model footprint exceeded memory budget");
        rejects([] { choose_context(32768, 1024 * mib, 256 * mib, 1048576); }, "minimum context exceeded RAM");
        rejects([] { choose_context(256, 4096 * mib, 512 * mib, 65536); }, "undersized training context accepted");
        require(reply_budget(512) == 128 && reply_budget(8192) == 512, "reply budget policy failed");
        rejects([] { pocket::Engine engine({.context = -1}); }, "negative context accepted");
        rejects([] { pocket::Engine engine({.threads = 0}); }, "zero threads accepted");
        rejects([] { pocket::Engine engine({.batch = -1}); }, "negative batch accepted");
        rejects([] { pocket::Engine engine({.gpu_layers = -1}); }, "negative GPU layers accepted");
        pocket::Engine engine;
        require(!engine.loaded(), "engine unexpectedly loaded");
        require(engine.config().context == 0 && engine.model_context() == 0, "automatic unloaded settings incorrect");
        rejects([&] { engine.generate({{"user", "Hello"}}, 8, 0, [](auto) { return true; }); },
                "unloaded generation accepted");
        rejects([&] { engine.load(""); }, "empty model path accepted");
        rejects([&] { engine.load("missing-pocket-test-model.gguf"); }, "missing model accepted");
        rejects([&] { engine.load(std::filesystem::temp_directory_path().string()); }, "directory model accepted");
        engine.unload();
        engine.unload();
        require(engine.model_path().empty(), "unload retained model path");

        if (argc > 1) {
            engine.load(argv[1]);
            require(engine.loaded(), "model not loaded");
            require(engine.config().context >= 512 && engine.config().context <= 8192,
                    "automatic context outside application limits");
            require(engine.config().context <= engine.model_context(), "automatic context exceeds training limit");
            const std::vector<pocket::Message> messages{{"user", "Reply with just the word Hello. /no_think"}};
            std::string first;
            auto result = engine.generate(messages, 16, 0, [&](auto text) { first += text; return true; });
            require(result.generated_tokens > 0 && result.generated_tokens <= 16, "invalid token count");
            require(!first.empty(), "no model text");
            std::string second;
            engine.generate(messages, 16, 0, [&](auto text) { second += text; return true; });
            require(first == second, "greedy generation changed between independent sessions");
            result = engine.generate(messages, 16, 0, [](auto) { return false; });
            require(result.generated_tokens == 0, "prefill cancellation failed");
            require(result.context == engine.config().context, "result context is incorrect");
            auto automatic = engine.generate(messages, 0, 0, [](auto) { return false; });
            require(automatic.history_trimmed == 0, "short automatic request was trimmed");
            std::string near_limit;
            for (int i = 0; i < engine.config().context - 64; ++i) near_limit += " a";
            const auto tight = engine.generate({{"user", near_limit}}, 0, 0, [](auto) { return false; });
            require(tight.prompt_tokens < tight.context &&
                    tight.prompt_tokens > tight.context - reply_budget(tight.context),
                    "near-limit prompt did not exercise reduced reply allowance");
            require(tight.history_trimmed == 0, "near-limit latest user was trimmed");
            const std::string oversized(200000, 'a');
            const std::vector<pocket::Message> long_history{
                {"system", "Answer briefly."}, {"user", oversized}, {"assistant", "Understood."},
                {"user", "Hello. /no_think"}};
            const auto trimmed = engine.generate(long_history, 0, 0, [](auto) { return false; });
            require(trimmed.history_trimmed == 2, "automatic history did not remove one complete turn");
            const auto preserved = engine.generate({long_history.front(), long_history.back()}, 0, 0,
                                                   [](auto) { return false; });
            require(trimmed.prompt_tokens == preserved.prompt_tokens, "system or latest user was not preserved");
            rejects([&] { engine.generate(long_history, 16, 0, [](auto) { return false; }); },
                    "explicit output silently trimmed history");
            rejects([&] { engine.generate({{"user", oversized}}, 0, 0, [](auto) { return false; }); },
                    "automatic output silently truncated latest user");
            rejects([&] { engine.generate({{"user", std::string(100000, 'a')}}, 16, 0,
                                          [](auto) { return true; }); }, "oversized context accepted");
            rejects([&] { engine.generate({{"tool", "bad role"}}, 16, 0,
                                          [](auto) { return true; }); }, "unsupported role accepted");
            rejects([&] { engine.generate(messages, 16, -1,
                                          [](auto) { return true; }); }, "negative temperature accepted");
            engine.unload();
            require(!engine.loaded(), "model remained loaded");
            require(engine.config().context == 0 && engine.model_context() == 0, "unload retained effective context");
            engine.load(argv[1]);
            require(engine.loaded(), "model reload failed");
        }
        std::cout << "Runtime checks passed" << (argc > 1 ? " (including real model)\n" : " (no model supplied)\n");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime test failed: " << error.what() << '\n';
        return 1;
    }
}
