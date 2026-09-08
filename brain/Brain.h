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

    std::vector<Synapse*> allSynapses;

    std::vector<std::unique_ptr<Synapse>> synapses;

    double learningRate = 0.02;
    double dopamine = 0.0;
    double dopamineDecay = 0.8;

    int episodesTrained = 0;
    int successes = 0;

    double totalReward = 0.0;

    long totalSteps = 0;
    long totalFired = 0;

    double bestReward = 0.0;

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

    void getAllNeurons(std::vector<Neuron*>& out);
    void getAllNeuronsConst(std::vector<const Neuron*>& out) const;

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

    void tick();
    void setInput(int index, double value);
    double getOutput(int index) const;
    void resetOutputs();
    void reward(double amount);
    void releaseDopamine(double amount);
    void adjustThresholds(int targetRate);
    void reset();

    std::vector<int> getLayerSizes() const;
    std::vector<int> getWiringLimits() const;

    std::vector<double> getNeuronThresholds() const;
    std::vector<double> getSynapseStrengths() const;

    void setNeuronThresholds(
        const std::vector<double>& thresholds
    );

    void setSynapseStrengths(
        const std::vector<double>& strengths
    );

    double getLearningRate() const;
    void setLearningRate(double value);

    int getEpisodesTrained() const;
    int getSuccesses() const;
    double getTotalReward() const;
    long getTotalSteps() const;
    long getTotalFired() const;
    double getBestReward() const;

    void setEpisodesTrained(int value);
    void setSuccesses(int value);
    void setTotalReward(double value);
    void setTotalSteps(long value);
    void setTotalFired(long value);
    void setBestReward(double value);
};

#endif
