// -----------------------------------------------------------------------------
// aeon_serve — the composition root of the G5 server.
//
// This is the one place the engine (G4) and the serving layer (G5) are wired
// together, because naming `V4Engine` is exactly what the serving layer is not
// allowed to do. It starts HTTP first (so `/health` answers during the multi-second
// load), builds the engine on the main thread, and drives the conversation on that
// same thread — HIP's device context is per-thread, so nothing else may touch it.
// -----------------------------------------------------------------------------

#include "tools/engine_cli.hpp"

#include "architecture/deepseek_v4/runtime/v4_conversation.hpp"
#include "architecture/deepseek_v4/runtime/v4_engine.hpp"
#include "platform/device.hpp"
#include "server/conversation_service.hpp"
#include "server/http_server.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

volatile std::sig_atomic_t g_stop = 0;

extern "C" void handle_signal(int) { g_stop = 1; }

struct ServeOptions {
    aeon::tools::EngineCli engine;
    std::string host{"127.0.0.1"};
    int port{8080};
    size_t max_queue{16};
    std::string model_name;
};

void print_usage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "Options:\n"
        << "  --model-dir <path>       Native Aeon model directory\n"
        << "  --context-size <n>       Context capacity in tokens (default: 4096)\n"
        << "  --warm-gib <n>           Warm host allocation in GiB (default: 0)\n"
        << "  --no-warm-preload        Allocate Warm capacity without startup payload reads\n"
        << "  --prefill-window <n>     Layer-major prefill window W in tokens\n"
        << "  --prefill-chunk <n>      Body chunk C in tokens, 1..64 (default: 64)\n"
        << "  --prefill-sweep-min-tokens <n>\n"
        << "                           Prompt length at or above which the sweep supplies experts\n"
        << "  --staging-blocks <n>     Swept-prefill staging arena in layer-blocks (default: 2)\n"
        << "  --verbose                Print the memory budget and residency summary\n"
        << "  --host <addr>            Bind address (default: 127.0.0.1)\n"
        << "  --port <n>               Bind port (default: 8080, 0 = kernel-chosen)\n"
        << "  --max-queue <n>          Queue capacity before requests are refused (default: 16)\n"
        << "  --model-name <id>        Model id reported by /v1/models and /status\n"
        << "  --help                   Show this help\n";
}

ServeOptions parse_options(int argc, char** argv) {
    ServeOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (aeon::tools::parse_engine_flag(options.engine, argc, argv, index)) continue;

        if (argument == "--host") {
            options.host = aeon::tools::engine_cli_value(argc, argv, index, "--host");
        } else if (argument == "--port") {
            options.port = static_cast<int>(
                aeon::tools::engine_cli_unsigned(
                    aeon::tools::engine_cli_value(argc, argv, index, "--port"), "--port"));
        } else if (argument == "--max-queue") {
            options.max_queue = static_cast<size_t>(aeon::tools::engine_cli_unsigned(
                aeon::tools::engine_cli_value(argc, argv, index, "--max-queue"), "--max-queue"));
        } else if (argument == "--model-name") {
            options.model_name = aeon::tools::engine_cli_value(argc, argv, index, "--model-name");
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.max_queue == 0) throw std::runtime_error("--max-queue must be positive");
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ServeOptions options = parse_options(argc, argv);

        aeon::core::select_compute_device(true);

        aeon::server::ConversationService service(options.max_queue);
        aeon::server::HttpOptions http_options;
        http_options.host = options.host;
        http_options.port = options.port;
        http_options.max_queue = options.max_queue;

        auto http = std::make_unique<aeon::server::HttpServer>(service, http_options);
        http->start();
        std::cout << "[aeon_serve] listening on " << options.host << ":" << http->port()
                  << " (loading model)" << std::endl;

        aeon::core::V4Engine engine;
        try {
            aeon::core::V4EngineOptions engine_options = options.engine.to_engine_options();
            engine.initialize(engine_options);
        } catch (...) {
            http->stop();
            throw;
        }

        const std::string model_id =
            options.model_name.empty() ? std::string("deepseek-v4-flash") : options.model_name;
        aeon::core::V4Conversation conversation(engine, model_id);

        std::cout << "[aeon_serve] model=" << model_id
                  << " context=" << engine.host().context_capacity()
                  << " queue=" << options.max_queue << std::endl;

        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::thread watcher([&] {
            while (g_stop == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            service.shutdown();
            http->stop();
        });

        // Blocks until `shutdown`. The engine thread is this one.
        service.run_worker(conversation);

        g_stop = 1;
        watcher.join();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[aeon_serve] error: " << error.what() << std::endl;
        return 1;
    }
}
