#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <filesystem>

#include "brain/Brain.h"
#include "utils/BrainSave.h"

static int testCount = 0;
static int passCount = 0;

#define CHECK(cond, msg) \
    do { \
        testCount++; \
        if (cond) { passCount++; std::cout << "  [PASS] " << msg << "\n"; } \
        else { std::cout << "  [FAIL] " << msg << "  (line " << __LINE__ << ")\n"; } \
    } while(0)

void testTick() {
    std::cout << "Test: tick() advances network\n";
    Brain brain(4, 8, 8, 8, 2, 3, 3, 3, 2);
    brain.setInput(0, 5.0);
    brain.tick();
    CHECK(brain.getTotalSteps() == 1, "totalSteps incremented after tick");
    CHECK(brain.getOutput(0) >= 0.0, "output voltage is read-able");
}

void testSetInputGetOutput() {
    std::cout << "Test: setInput/getOutput\n";
    Brain brain(4, 8, 8, 8, 2, 3, 3, 3, 2);
    brain.setInput(0, 10.0);
    CHECK(true, "setInput(0) does not throw");
    double out = brain.getOutput(0);
    CHECK(out >= 0.0, "getOutput returns non-negative value");
}

void testReward() {
    std::cout << "Test: reward() updates totalReward\n";
    Brain brain(4, 8, 8, 8, 2, 3, 3, 3, 2);
    brain.reward(1.5);
    brain.reward(-0.6);
    CHECK(std::abs(brain.getTotalReward() - 0.9) < 1e-9, "totalReward accumulates (1.5 - 0.6 = 0.9)");
}

void testSaveLoad() {
    std::cout << "Test: BrainSave roundtrip\n";
    Brain brain(4, 8, 8, 8, 2, 3, 3, 3, 2);
    brain.setLearningRate(0.05);

    auto dir = std::filesystem::temp_directory_path() / "sage_test";
    auto file = BrainSave::save(brain, dir);
    Brain loaded = BrainSave::load(file);
    CHECK(loaded.getLearningRate() == 0.05, "learningRate preserved");
    CHECK(loaded.getLayerSizes() == brain.getLayerSizes(), "topology preserved");
    CHECK(loaded.getWiringLimits() == brain.getWiringLimits(), "wiring preserved");

    auto th1 = brain.getNeuronThresholds();
    auto th2 = loaded.getNeuronThresholds();
    bool same = (th1.size() == th2.size());
    for (size_t i = 0; same && i < th1.size(); i++) {
        if (std::abs(th1[i] - th2[i]) > 1e-12) same = false;
    }
    CHECK(same, "thresholds preserved");

    auto s1 = brain.getSynapseStrengths();
    auto s2 = loaded.getSynapseStrengths();
    same = (s1.size() == s2.size());
    for (size_t i = 0; same && i < s1.size(); i++) {
        if (std::abs(s1[i] - s2[i]) > 1e-12) same = false;
    }
    CHECK(same, "synapse strengths preserved");
}

void testReset() {
    std::cout << "Test: reset() clears transient state\n";
    Brain brain(4, 8, 8, 8, 2, 3, 3, 3, 2);
    brain.setInput(0, 5.0);
    brain.tick();
    brain.reset();
    CHECK(std::abs(brain.getTotalReward()) < 1e-12, "reward cleared on reset");
}

int main() {
    std::cout << "Sage Brain Engine Tests\n";
    std::cout << "=======================\n\n";

    testTick();
    testSetInputGetOutput();
    testReward();
    testSaveLoad();
    testReset();

    std::cout << "\n=======================\n";
    std::cout << passCount << "/" << testCount << " tests passed\n";
    std::cout << "=======================\n";

    return (passCount == testCount) ? 0 : 1;
}
