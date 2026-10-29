#include "pocket/engine.hpp"
#include "httplib.h"
#include "json.hpp"
#include <filesystem>
#include <iostream>
#include <mutex>
#include <charconv>
#include <memory>
#include <cmath>

using json = nlohmann::json;
namespace fs = std::filesystem;

static void reply(httplib::Response& res, const json& value, int status = 200) {
    res.status = status;
    res.set_content(value.dump(), "application/json");
}
static int number(const std::string& value) {
    int result{};
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument("Invalid integer: " + value);
    return result;
}
int main(int argc, char** argv) try {
    pocket::Config config;
    fs::path root = "models", web = POCKET_WEB_DIR;
    int port = 8080;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            std::cout << "Pocket Inference\n  --models DIR --web DIR --port 8080\n  --context 2048 --threads 2 --batch 128 --gpu-layers 0\nLocal UI: http://127.0.0.1:8080\n";
            return 0;
        }
        if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--models") root = value;
        else if (option == "--web") web = value;
        else if (option == "--port") port = number(value);
        else if (option == "--context") config.context = number(value);
        else if (option == "--threads") config.threads = number(value);
        else if (option == "--batch") config.batch = number(value);
        else if (option == "--gpu-layers") config.gpu_layers = number(value);
        else throw std::invalid_argument("Unknown option: " + option);
    }
    if (port < 1 || port > 65535) throw std::invalid_argument("Port out of range");
    fs::create_directories(root);
    root = fs::canonical(root);
    pocket::Engine engine(config);
    std::mutex mutex;
    httplib::Server server;
    server.new_task_queue = [] { return new httplib::ThreadPool(4, 8); };
    server.set_payload_max_length(1024 * 1024);
    server.set_read_timeout(10);
    server.set_write_timeout(10);
    server.set_keep_alive_max_count(10);
    server.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'");
        const auto host = req.get_header_value("Host");
        const auto local = "127.0.0.1:" + std::to_string(port);
        const auto localhost = "localhost:" + std::to_string(port);
        if (host != local && host != localhost) {
            reply(res, {{"error", "Invalid Host"}}, 403);
            return httplib::Server::HandlerResponse::Handled;
        }
        const auto origin = req.get_header_value("Origin");
        if ((!origin.empty() && origin != "http://" + host) ||
            (req.method == "POST" && req.get_header_value("Content-Type").find("application/json") != 0)) {
            reply(res, {{"error", "Same-origin JSON requests required"}}, 403);
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    server.set_exception_handler([](const auto&, auto& res, std::exception_ptr ep) {
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { reply(res, {{"error", e.what()}}, 400); }
    });
    const auto model_file = [&](const std::string& path) {
        if (path.empty() || path.find('\0') != std::string::npos || fs::path(path).is_absolute())
            throw std::invalid_argument("Choose a relative GGUF path inside the models directory");
        const auto resolved = fs::canonical(root / path);
        auto relative = resolved.lexically_relative(root);
        if (relative.empty() || *relative.begin() == ".." || resolved.extension() != ".gguf" || !fs::is_regular_file(resolved))
            throw std::invalid_argument("Model must be a GGUF file inside the models directory");
        return resolved;
    };
    server.Get("/api/status", [&](const auto&, auto& res) {
        std::unique_lock lock(mutex, std::try_to_lock);
        if (!lock.owns_lock()) { reply(res, {{"error", "Inference is busy"}}, 409); return; }
        reply(res, {{"loaded",engine.loaded()}, {"model",engine.model_path()}, {"context",config.context}, {"threads",config.threads}, {"batch",config.batch}});
    });
    server.Get("/api/models", [&](const auto&, auto& res) {
        auto models = json::array();
        for (const auto& entry : fs::directory_iterator(root)) {
            if (models.size() >= 1000) break;
            if (entry.path().extension() != ".gguf") continue;
            try {
                auto path = model_file(entry.path().filename().string());
                models.push_back({{"name",entry.path().filename().string()}, {"path",entry.path().filename().string()}, {"size_bytes",fs::file_size(path)}});
            } catch (const std::exception&) { /* Skip inaccessible or out-of-root symlinks. */ }
        }
        reply(res, {{"models",models}});
    });
    server.Post("/api/load", [&](const auto& req, auto& res) {
        std::unique_lock lock(mutex, std::try_to_lock);
        if (!lock.owns_lock()) { reply(res, {{"error","Inference is busy"}},409); return; }
        auto path = model_file(json::parse(req.body).at("path").template get<std::string>());
        engine.load(path.string());
        reply(res, {{"loaded",true},{"model",path.string()}});
    });
    server.Post("/api/unload", [&](const auto&, auto& res) {
        std::unique_lock lock(mutex, std::try_to_lock);
        if (!lock.owns_lock()) { reply(res, {{"error","Inference is busy"}},409); return; }
        engine.unload(); reply(res, {{"loaded",false}});
    });
    server.Post("/api/chat", [&](const httplib::Request& req, httplib::Response& res) {
        auto lock = std::make_shared<std::unique_lock<std::mutex>>(mutex, std::try_to_lock);
        if (!lock->owns_lock()) { reply(res, {{"error","Inference is busy"}},409); return; }
        if (!engine.loaded()) { reply(res, {{"error","Load a model first"}},400); return; }
        auto body = json::parse(req.body);
        std::vector<pocket::Message> messages;
        const auto& input = body.at("messages");
        if (!input.is_array() || input.empty() || input.size() > 256) throw std::invalid_argument("Supply 1 to 256 messages");
        for (const auto& message : input)
            messages.push_back({message.at("role").get<std::string>(),message.at("content").get<std::string>()});
        const auto limit = body.value("max_tokens", json(256));
        if (!limit.is_number_integer() || limit < 1 || limit > config.context)
            throw std::invalid_argument("max_tokens must be an integer within the context limit");
        const int max_tokens = limit.get<int>();
        const auto temp = body.value("temperature", json(0.7));
        if (!temp.is_number() || temp < 0 || temp > 2)
            throw std::invalid_argument("temperature must be between 0 and 2");
        const float temperature = temp.get<float>();
        res.set_header("Cache-Control", "no-store");
        res.set_chunked_content_provider("text/event-stream", [&, lock, messages = std::move(messages), max_tokens, temperature](size_t, httplib::DataSink& sink) {
            const auto send = [&](const json& event) { const auto data = "data: " + event.dump() + "\n\n"; return sink.write(data.data(),data.size()); };
            try {
                const auto result = engine.generate(messages,max_tokens,temperature,[&](std::string_view token) {
                    if (!sink.is_writable()) return false;
                    return token.empty() ? sink.write(": ping\n\n", 8) : send({{"token",token}});
                });
                send({{"done",true},{"prompt_tokens",result.prompt_tokens},{"generated_tokens",result.generated_tokens},{"seconds",result.seconds}});
            } catch(const std::exception& e) { send({{"error",e.what()}}); }
            sink.done();
            return true;
        });
    });
    if (!server.set_mount_point("/", web.string())) throw std::runtime_error("Web directory unavailable; set --web DIR");
    std::cout << "Pocket Inference: http://127.0.0.1:" << port << "\nModels: " << root << std::endl;
    if (!server.listen("127.0.0.1",port)) throw std::runtime_error("Could not bind local port");
    return 0;
} catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
