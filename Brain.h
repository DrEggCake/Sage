#ifndef BRAIN_H
#define BRAIN_H

#include <vector>
#include <memory>

#include "Neuron.h"
#include "Synapse.h"

class Brain {

private:

    int INPUT_NEURONS;
    int LAYER_1_NEURONS;
    int LAYER_2_NEURONS;
    int LAYER_3_NEURONS;
    int OUTPUT_NEURONS;

    int INPUT_TO_LAYER_1_LIMIT;
    int LAYER_1_TO_LAYER_2_LIMIT;
    int LAYER_2_TO_LAYER_3_LIMIT;
    int LAYER_3_TO_OUTPUT_LIMIT;

    std::vector<std::unique_ptr<Neuron>> inputNeurons;
    std::vector<std::unique_ptr<Neuron>> layer1Neurons;
    std::vector<std::unique_ptr<Neuron>> layer2Neurons;
    std::vector<std::unique_ptr<Neuron>> layer3Neurons;
    std::vector<std::unique_ptr<Neuron>> outputNeurons;

    std::vector<std::unique_ptr<Synapse>> synapses;

    double LEARNING_RATE = 0.02;
    double ELIGIBILITY_DECAY = 0.95;
    double LEAK_RATE = 0.0;

    int episodesTrained = 0;
    int successes = 0;

    double totalReward = 0.0;

    long totalSteps = 0;
    long totalFired = 0;

    double bestReward = 0.0;

    static constexpr double MIN_STRENGTH = -3.0;
    static constexpr double MAX_STRENGTH = 3.0;

    void buildNeurons();
    void buildConnections();

    void connectSliding(
        const std::vector<std::unique_ptr<Neuron>>& fromLayer,
        const std::vector<std::unique_ptr<Neuron>>& toLayer,
        int limit
    );

    void connect(
        Neuron* from,
        Neuron* to
    );

public:

    Brain();

    Brain(
        int inputNeurons,
        int layer1Neurons,
        int layer2Neurons,
        int layer3Neurons,
        int outputNeurons,
        int inputToLayer1,
        int layer1ToLayer2,
        int layer2ToLayer3,
        int layer3ToOutput
    );
};

#endif
