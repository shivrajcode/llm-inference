#include "pocket/engine.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

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
        rejects([] { pocket::Engine engine({.context = 0}); }, "zero context accepted");
        rejects([] { pocket::Engine engine({.threads = 0}); }, "zero threads accepted");
        rejects([] { pocket::Engine engine({.batch = -1}); }, "negative batch accepted");
        rejects([] { pocket::Engine engine({.gpu_layers = -1}); }, "negative GPU layers accepted");
        pocket::Engine engine;
        require(!engine.loaded(), "engine unexpectedly loaded");
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
            rejects([&] { engine.generate({{"user", std::string(100000, 'a')}}, 16, 0,
                                          [](auto) { return true; }); }, "oversized context accepted");
            rejects([&] { engine.generate({{"tool", "bad role"}}, 16, 0,
                                          [](auto) { return true; }); }, "unsupported role accepted");
            rejects([&] { engine.generate(messages, 16, -1,
                                          [](auto) { return true; }); }, "negative temperature accepted");
            engine.unload();
            require(!engine.loaded(), "model remained loaded");
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
