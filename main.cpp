#include <iostream>
#include <vector>
#include <random>
#include <filesystem>

#include "brain/Brain.h"
#include "utils/BrainSave.h"

// Demo: demonstrate the Sage SNN brain engine (ported from Sage-Java) by
// running a tiny reward-driven learning loop: pick the largest output after
// a fixed input, reward good choices, save the trained brain.
int main() {

    Brain brain(
        8,   // input neurons
        16,  // layer 1
        16,  // layer 2
        16,  // layer 3
        4,   // output neurons
        3,   // input -> layer 1
        3,   // layer 1 -> layer 2
        3,   // layer 2 -> layer 3
        3    // layer 3 -> output
    );

    std::cout << "Sage SNN brain created with "
              << brain.getLayerSizes()[0] << " inputs, "
              << brain.getLayerSizes()[4] << " outputs\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr int episodes = 200;
    constexpr int decisionTicks = 4;

    for (int ep = 0; ep < episodes; ep++) {

        brain.reset();
        brain.resetOutputs();

        for (int i = 0; i < 8; i++) {
            brain.setInput(i, dist(rng) * 2.0);
        }

        for (int t = 0; t < decisionTicks; t++) {
            brain.tick();
        }

        int best = 0;
        for (int o = 1; o < 4; o++) {
            if (brain.getOutput(o) > brain.getOutput(best)) {
                best = o;
            }
        }

        double reward = (best == 0) ? 1.0 : -0.5;
        brain.reward(reward);

        brain.adjustThresholds(2);
    }

    std::cout << "Trained " << episodes
              << " episodes (reward " << brain.getTotalReward() << ")\n";

    std::filesystem::create_directories("data/brain");
    auto file = BrainSave::save(brain, "data/brain");

    std::cout << "Saved brain to " << file << "\n";

    auto loaded = BrainSave::load(file);
    std::cout << "Loaded brain back with "
              << loaded.getLayerSizes()[0] << " inputs, "
              << loaded.getLayerSizes()[4] << " outputs — topology intact\n";

    return 0;
}
