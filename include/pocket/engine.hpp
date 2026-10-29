#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace pocket {

struct Config {
    int context = 2048;
    int threads = 2;
    int batch = 128;
    int gpu_layers = 0;
};

struct Message {
    std::string role;
    std::string content;
};

struct Result {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double seconds = 0;
};

// Calls must be serialized by the owner. Each generation has an independent KV
// cache; conversation history is explicitly supplied by the caller.
class Engine {
public:
    explicit Engine(Config config = {});
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    void load(std::string path);
    void unload();
    [[nodiscard]] bool loaded() const;
    [[nodiscard]] std::string model_path() const;
    Result generate(const std::vector<Message>& messages, int max_tokens,
                    float temperature,
                    const std::function<bool(std::string_view)>& emit);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pocket
