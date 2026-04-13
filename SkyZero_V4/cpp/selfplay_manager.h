#ifndef SKYZERO_SELFPLAY_MANAGER_H
#define SKYZERO_SELFPLAY_MANAGER_H

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <iostream>

namespace skyzero {

// Watches a directory for the latest TorchScript .pt model file.
class SelfplayManager {
public:
    explicit SelfplayManager(const std::string& model_dir)
        : model_dir_(model_dir) {}

    // Returns path to the latest .pt file, or empty string if none found.
    std::string get_latest_model() const {
        namespace fs = std::filesystem;
        if (!fs::exists(model_dir_)) return "";

        std::string best_path;
        fs::file_time_type best_time{};
        bool found = false;

        for (const auto& entry : fs::directory_iterator(model_dir_)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".pt") continue;

            const auto t = entry.last_write_time();
            if (!found || t > best_time) {
                found = true;
                best_time = t;
                best_path = entry.path().string();
            }
        }
        return best_path;
    }

    // Blocks until a .pt model is available in the directory.
    std::string wait_for_model(int poll_interval_ms = 5000) const {
        std::cout << "Waiting for model in " << model_dir_ << "..." << std::endl;
        while (true) {
            auto path = get_latest_model();
            if (!path.empty()) {
                std::cout << "Found model: " << path << std::endl;
                return path;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
        }
    }

    // Check if a newer model is available than the given path.
    bool has_newer_model(const std::string& current_path) const {
        auto latest = get_latest_model();
        return !latest.empty() && latest != current_path;
    }

private:
    std::string model_dir_;
};

}  // namespace skyzero

#endif  // SKYZERO_SELFPLAY_MANAGER_H
