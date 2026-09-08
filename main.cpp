#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <fstream>
#include <filesystem>

#include "brain/Brain.h"
#include "utils/BrainSave.h"

// Demo + training: drive the Sage SNN brain (ported from Sage-Java) with a
// simple reward task (pick output 0 on random input) under epsilon-greedy
// exploration, then save the trained brain for later replay/inspection.
int main(int argc, char** argv) {

    int episodes = 200;
    int decisionTicks = 4;
    double epsilonStart = 1.0;
    double epsilonEnd = 0.05;
    double learningRate = 0.02;
    unsigned seed = 42;
    std::string fireLog;
    std::string saveDir = "data/brain";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto hasValue = i + 1 < argc;
        auto parseDouble = [&](const std::string& s) {
            try { return std::stod(s); }
            catch (...) { return 0.0; }
        };
        if (a == "--episodes" && hasValue) episodes = (int)parseDouble(argv[++i]);
        else if (a == "--decision-ticks" && hasValue) decisionTicks = (int)parseDouble(argv[++i]);
        else if (a == "--epsilon-start" && hasValue) epsilonStart = parseDouble(argv[++i]);
        else if (a == "--epsilon-end" && hasValue) epsilonEnd = parseDouble(argv[++i]);
        else if (a == "--learning-rate" && hasValue) learningRate = parseDouble(argv[++i]);
        else if (a == "--seed" && hasValue) seed = (unsigned)parseDouble(argv[++i]);
        else if (a == "--fire-log" && hasValue) fireLog = argv[++i];
        else if (a == "--save-dir" && hasValue) saveDir = argv[++i];
        else if (a == "--help") {
            std::cout << "Usage: sage [--episodes N] [--decision-ticks N]\n"
                      << "            [--epsilon-start E] [--epsilon-end E]\n"
                      << "            [--learning-rate R] [--seed S]\n"
                      << "            [--fire-log FILE] [--save-dir DIR]\n";
            return 0;
        }
    }

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

    brain.setLearningRate(learningRate);

    std::cout << "Sage SNN training\n"
              << "  topology: " << brain.getLayerSizes()[0]
              << " -> " << brain.getLayerSizes()[1]
              << " -> " << brain.getLayerSizes()[2]
              << " -> " << brain.getLayerSizes()[3]
              << " -> " << brain.getLayerSizes()[4] << "\n"
              << "  episodes: " << episodes
              << ", decision ticks/ep: " << decisionTicks
              << ", lr: " << learningRate
              << ", seed: " << seed << "\n"
              << "  epsilon: " << epsilonStart << " -> " << epsilonEnd
              << " (linear decay)\n";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    double totalRewardAcc = 0.0;
    int successes = 0;
    double bestEpisodeReward = -std::numeric_limits<double>::infinity();
    int reportEvery = std::max(1, episodes / 10);

    std::vector<int> topology = brain.getLayerSizes();
    std::ofstream logger;
    if (!fireLog.empty()) {
        auto logPath = std::filesystem::path(fireLog);
        if (logPath.has_parent_path()) {
            std::filesystem::create_directories(logPath.parent_path());
        }
        logger.open(fireLog);
        logger << "# topology " << topology[0] << ',' << topology[1] << ','
               << topology[2] << ',' << topology[3] << ',' << topology[4] << '\n';
        logger << "# episodes " << episodes << " decisionTicks " << decisionTicks
               << " seed " << seed << " lr " << learningRate
               << " epsilonStart " << epsilonStart << " epsilonEnd " << epsilonEnd << '\n';
        logger << "ep,tick";
        for (int i = 0; i < topology[0]; i++) logger << ",i" << i;
        int totalNeurons = 0;
        for (int s : topology) totalNeurons += s;
        for (int i = 0; i < totalNeurons; i++) logger << ",f" << i;
        logger << '\n';
    }

    for (int ep = 0; ep < episodes; ep++) {

        double eps = epsilonEnd;
        if (episodes > 1) {
            eps = epsilonStart + (epsilonEnd - epsilonStart) *
                                 ((double)ep / (double)(episodes - 1));
        }

        brain.reset();
        brain.resetOutputs();

        std::vector<double> inputs(topology[0]);
        for (int i = 0; i < topology[0]; i++) {
            double v = dist(rng) * 2.0;
            inputs[i] = v;
            brain.setInput(i, v);
        }

        for (int t = 0; t < decisionTicks; t++) {
            brain.tick();
            if (logger.is_open()) {
                logger << ep << ',' << t;
                for (double v : inputs) logger << ',' << v;
                for (int f : brain.getFiredFlags()) logger << ',' << f;
                logger << '\n';
            }
        }

        int best = 0;
        for (int o = 1; o < 4; o++) {
            if (brain.getOutput(o) > brain.getOutput(best)) {
                best = o;
            }
        }

        int chosen = (coin(rng) < eps) ? (int)(coin(rng) * 4.0) % 4 : best;
        double reward = (chosen == 0) ? 1.0 : -0.5;

        brain.reward(reward);
        totalRewardAcc += reward;
        if (chosen == 0) successes++;
        bestEpisodeReward = std::max(bestEpisodeReward, reward);

        brain.adjustThresholds(2);

        if ((ep == 0) || ((ep + 1) % reportEvery == 0)) {
            std::cout << "  ep " << (ep + 1) << "/" << episodes
                      << "  epsilon " << round(eps * 1000) / 1000
                      << "  correct " << successes << "/" << (ep + 1)
                      << "  reward+ " << round(totalRewardAcc * 1000) / 1000
                      << "\n";
        }
    }

    double accuracy = (double)successes / (double)episodes * 100.0;

    std::cout << "Done: " << episodes << " eps, "
              << successes << " correct (" << round(accuracy * 10) / 10
              << "%), total reward " << round(totalRewardAcc * 1000) / 1000
              << ", seed " << seed << "\n";

    brain.setEpisodesTrained(episodes);
    brain.setSuccesses(successes);
    brain.setTotalReward(totalRewardAcc);
    brain.setBestReward(bestEpisodeReward);

    std::filesystem::create_directories(saveDir);
    auto file = BrainSave::save(brain, saveDir);

    std::cout << "Saved brain to " << file << "\n";

    if (!fireLog.empty()) {
        std::string metaPath = fireLog + ".meta.json";
        std::ofstream meta(metaPath);
        meta << "{\n"
             << "  \"brain\": \"" << file.filename().string() << "\",\n"
             << "  \"layerSizes\": [" << topology[0] << ',' << topology[1] << ','
             << topology[2] << ',' << topology[3] << ',' << topology[4] << "],\n"
             << "  \"wiringLimits\": [" << brain.getWiringLimits()[0] << ',' << brain.getWiringLimits()[1]
             << ',' << brain.getWiringLimits()[2] << ',' << brain.getWiringLimits()[3] << "],\n"
             << "  \"episodes\": " << episodes << ",\n"
             << "  \"decisionTicks\": " << decisionTicks << ",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"learningRate\": " << learningRate << ",\n"
             << "  \"epsilonStart\": " << epsilonStart << ",\n"
             << "  \"epsilonEnd\": " << epsilonEnd << ",\n"
             << "  \"successes\": " << successes << ",\n"
             << "  \"totalReward\": " << totalRewardAcc << ",\n"
             << "  \"episodesTrained\": " << episodes << "\n"
             << "}\n";
        std::cout << "Fire log: " << fireLog << " (+ sidecar " << metaPath << ")\n";
    }

    auto loaded = BrainSave::load(file);
    std::cout << "Loaded brain back with "
              << loaded.getLayerSizes()[0] << " inputs, "
              << loaded.getLayerSizes()[4] << " outputs — topology intact\n";

    return 0;
}