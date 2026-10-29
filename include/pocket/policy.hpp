#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pocket::policy {
inline constexpr std::uint64_t mib = 1024 * 1024;

// Conservative fallback for models whose scalar attention metadata is absent.
inline constexpr std::uint64_t fallback_kv_bytes_per_token = 256 * 1024;

inline std::uint64_t kv_bytes_per_token(std::uint64_t layers, std::uint64_t kv_heads,
                                        std::uint64_t key_length, std::uint64_t value_length) {
    if (!layers || !kv_heads || !key_length || !value_length)
        return fallback_kv_bytes_per_token;
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    if (key_length > maximum - value_length) return maximum;
    auto value = key_length + value_length;
    for (auto factor : {layers, kv_heads, std::uint64_t{2}}) {
        if (value > maximum / factor) return maximum;
        value *= factor;
    }
    return value; // Two-byte F16 keys and values.
}

inline int choose_context(int model_context, std::uint64_t available_bytes,
                          std::uint64_t model_bytes, std::uint64_t kv_bytes) {
    if (model_context < 512)
        throw std::invalid_argument("Model context is too small for automatic chat (minimum 512)");
    // Unknown host memory is treated conservatively as 1 GiB available.
    if (!available_bytes) available_bytes = 1024 * mib;
    const auto reserve = std::max(512 * mib, available_bytes / 4);
    if (model_bytes >= available_bytes || reserve >= available_bytes - model_bytes)
        throw std::runtime_error("Not enough available memory for this model; close other apps or choose a smaller GGUF");
    // Leave half the remainder for runtime buffers and measurement uncertainty.
    const auto kv_budget = (available_bytes - model_bytes - reserve) / 2;
    if (!kv_bytes) kv_bytes = fallback_kv_bytes_per_token;
    const auto affordable = kv_budget / kv_bytes;
    const auto ceiling = static_cast<std::uint64_t>(std::min(model_context, 8192));
    const auto chosen = std::min(ceiling, affordable) / 256 * 256;
    if (chosen < 512)
        throw std::runtime_error("Not enough memory for a minimum chat context; close other apps or choose a smaller GGUF");
    return static_cast<int>(chosen);
}

inline int reply_budget(int context) { return std::min(512, context / 4); }
} // namespace pocket::policy
