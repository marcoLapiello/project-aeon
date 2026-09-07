#include "core/config.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Loading DeepSeek-V4 config from JSON..." << std::endl;

    std::string config_path = "models/DeepSeek-V4-Flash-0731-INT4-W4A16/"
                              "models--yiminyuan--DeepSeek-V4-Flash-0731-INT4-W4A16/"
                              "snapshots/64700592cadaf205fe0c13202061ff4b45afbfd0/config.json";

    auto cfg = aeon::core::DeepSeekV4Config::load_from_json(config_path);

    std::cout << "Model Type: " << cfg.model_type << std::endl;
    std::cout << "Hidden Size: " << cfg.hidden_size << std::endl;
    std::cout << "MoE Intermediate Size: " << cfg.moe_intermediate_size << std::endl;
    std::cout << "Num Hidden Layers: " << cfg.num_hidden_layers << std::endl;
    std::cout << "Num Hash Layers: " << cfg.num_hash_layers << std::endl;
    std::cout << "Num Routed Experts: " << cfg.n_routed_experts << std::endl;
    std::cout << "Num Shared Experts: " << cfg.n_shared_experts << std::endl;
    std::cout << "Num Experts Per Tok: " << cfg.num_experts_per_tok << std::endl;
    std::cout << "SwiGLU Limit: " << cfg.swiglu_limit << std::endl;
    std::cout << "HC Mult: " << cfg.hc_mult << std::endl;
    std::cout << "HC Sinkhorn Iters: " << cfg.hc_sinkhorn_iters << std::endl;
    std::cout << "Quant Bits: " << cfg.quant.num_bits << std::endl;
    std::cout << "Quant Group Size: " << cfg.quant.group_size << std::endl;
    std::cout << "Quant Symmetric: " << (cfg.quant.symmetric ? "true" : "false") << std::endl;

    // Golden assertions
    assert(cfg.hidden_size == 4096);
    assert(cfg.moe_intermediate_size == 2048);
    assert(cfg.num_hidden_layers == 43);
    assert(cfg.num_hash_layers == 3);
    assert(cfg.n_routed_experts == 256);
    assert(cfg.n_shared_experts == 1);
    assert(cfg.num_experts_per_tok == 6);
    assert(cfg.swiglu_limit == 10.0f);
    assert(cfg.hc_mult == 4);
    assert(cfg.hc_sinkhorn_iters == 20);
    assert(cfg.sliding_window == 128);
    assert(cfg.scoring_func == "sqrtsoftplus");
    assert(cfg.quant.num_bits == 4);
    assert(cfg.quant.group_size == 32);
    assert(cfg.quant.symmetric == true);

    std::cout << "[Test PASS] DeepSeek-V4 config successfully parsed and verified!" << std::endl;
    return 0;
}
