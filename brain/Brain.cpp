#include "Brain.h"

#include <algorithm>


Brain::Brain() {

    INPUT_NEURONS = 0;
    LAYER_1_NEURONS = 0;
    LAYER_2_NEURONS = 0;
    LAYER_3_NEURONS = 0;
    OUTPUT_NEURONS = 0;

    INPUT_TO_LAYER_1_LIMIT = 0;
    LAYER_1_TO_LAYER_2_LIMIT = 0;
    LAYER_2_TO_LAYER_3_LIMIT = 0;
    LAYER_3_TO_OUTPUT_LIMIT = 0;
}


Brain::Brain(
    int inputNeurons,
    int layer1Neurons,
    int layer2Neurons,
    int layer3Neurons,
    int outputNeurons,
    int inputToLayer1,
    int layer1ToLayer2,
    int layer2ToLayer3,
    int layer3ToOutput
) {

    this->INPUT_NEURONS = inputNeurons;
    this->LAYER_1_NEURONS = layer1Neurons;
    this->LAYER_2_NEURONS = layer2Neurons;
    this->LAYER_3_NEURONS = layer3Neurons;
    this->OUTPUT_NEURONS = outputNeurons;

    this->INPUT_TO_LAYER_1_LIMIT = inputToLayer1;
    this->LAYER_1_TO_LAYER_2_LIMIT = layer1ToLayer2;
    this->LAYER_2_TO_LAYER_3_LIMIT = layer2ToLayer3;
    this->LAYER_3_TO_OUTPUT_LIMIT = layer3ToOutput;

    buildNeurons();
    buildConnections();
}


void Brain::buildNeurons() {

    for (int i = 0; i < INPUT_NEURONS; i++) {

        auto neuron =
            std::make_unique<Neuron>(
                Neuron::NeuronType::INPUT
            );

        inputNeurons.push_back(
            std::move(neuron)
        );
    }


    for (int i = 0; i < LAYER_1_NEURONS; i++) {

        auto neuron =
            std::make_unique<Neuron>(
                Neuron::NeuronType::INTERNAL
            );

        layer1Neurons.push_back(
            std::move(neuron)
        );
    }


    for (int i = 0; i < LAYER_2_NEURONS; i++) {

        auto neuron =
            std::make_unique<Neuron>(
                Neuron::NeuronType::INTERNAL
            );

        layer2Neurons.push_back(
            std::move(neuron)
        );
    }


    for (int i = 0; i < LAYER_3_NEURONS; i++) {

        auto neuron =
            std::make_unique<Neuron>(
                Neuron::NeuronType::INTERNAL
            );

        layer3Neurons.push_back(
            std::move(neuron)
        );
    }


    for (int i = 0; i < OUTPUT_NEURONS; i++) {

        auto neuron =
            std::make_unique<Neuron>(
                Neuron::NeuronType::OUTPUT
            );

        outputNeurons.push_back(
            std::move(neuron)
        );
    }
}


void Brain::buildConnections() {

    connectSliding(
        inputNeurons,
        layer1Neurons,
        INPUT_TO_LAYER_1_LIMIT
    );

    connectSliding(
        layer1Neurons,
        layer2Neurons,
        LAYER_1_TO_LAYER_2_LIMIT
    );

    connectSliding(
        layer2Neurons,
        layer3Neurons,
        LAYER_2_TO_LAYER_3_LIMIT
    );

    connectSliding(
        layer3Neurons,
        outputNeurons,
        LAYER_3_TO_OUTPUT_LIMIT
    );
}

/**
* Evenly spaced sliding window wiring.
* Source neuron i in a layer of size N1 connects to K target neurons in a layer of size N2,
* starting at floor(i * N2 - K) / (N1 - 1)).
* Source 0 always maps to targets 0..K-1 and source N1-1 maps to targets (N2 - K)..(N2 -1),
* so every target neuron receives at least one connection and every source neuron sends exactly K synapses. This prevents any dead neurons from forming.
*/
void Brain::connectSliding(
    const std::vector<std::unique_ptr<Neuron>>& fromLayer,
    const std::vector<std::unique_ptr<Neuron>>& toLayer,
    int limit
) {

    int N1 = fromLayer.size();
    int N2 = toLayer.size();

    int K = std::min(limit, N2);

    if (K <= 0 || N1 <= 0 || N2 <= 0) {
        return;
    }


    for (int i = 0; i < N1; i++) {

        int start =
            N1 <= 1
            ? 0
            : static_cast<int>(
                static_cast<long long>(i) *
                (N2 - K) /
                (N1 - 1)
            );


        for (int j = 0; j < K; j++) {

            connect(
                fromLayer[i].get(),
                toLayer[start + j].get()
            );
        }
    }
}


void Brain::connect(Neuron* from, Neuron* to) {

    auto synapse =
        std::make_unique<Synapse>(
            from,
            to
        );

    Synapse* synapsePtr = synapse.get();

    from->addSynapseOut(synapsePtr);
    to->addSynapseIn(synapsePtr);

    synapses.push_back(std::move(synapse));
}

// Serialization / persistence

std::vector<int> Brain::getLayerSizes() const {

    return {
        INPUT_NEURONS,
        LAYER_1_NEURONS,
        LAYER_2_NEURONS,
        LAYER_3_NEURONS,
        OUTPUT_NEURONS
    };
}


std::vector<int> Brain::getWiringLimits() const {

    return {
        INPUT_TO_LAYER_1_LIMIT,
        LAYER_1_TO_LAYER_2_LIMIT,
        LAYER_2_TO_LAYER_3_LIMIT,
        LAYER_3_TO_OUTPUT_LIMIT
    };
}


// Learning configuration

double Brain::getLearningRate() const {
    return LEARNING_RATE;
}


double Brain::getEligibilityDecay() const {
    return ELIGIBILITY_DECAY;
}


double Brain::getLeakRate() const {
    return LEAK_RATE;
}


void Brain::setLearningRate(double value) {
    LEARNING_RATE = value;
}


void Brain::setEligibilityDecay(double value) {
    ELIGIBILITY_DECAY = value;
}


void Brain::setLeakRate(double value) {
    LEAK_RATE = value;
}

// training statistics

int Brain::getEpisodesTrained() const {
    return episodesTrained;
}


int Brain::getSuccesses() const {
    return successes;
}


double Brain::getTotalReward() const {
    return totalReward;
}


long Brain::getTotalSteps() const {
    return totalSteps;
}


long Brain::getTotalFired() const {
    return totalFired;
}


double Brain::getBestReward() const {
    return bestReward;
}


void Brain::setEpisodesTrained(int value) {
    episodesTrained = value;
}


void Brain::setSuccesses(int value) {
    successes = value;
}


void Brain::setTotalReward(double value) {
    totalReward = value;
}


void Brain::setTotalSteps(long value) {
    totalSteps = value;
}


void Brain::setTotalFired(long value) {
    totalFired = value;
}


void Brain::setBestReward(double value) {
    bestReward = value;
}
// ============================================================
// Neuron serialization
// ============================================================

std::vector<double> Brain::getNeuronThresholds() const {

    std::vector<double> thresholds;

    thresholds.reserve(
        inputNeurons.size() +
        layer1Neurons.size() +
        layer2Neurons.size() +
        layer3Neurons.size() +
        outputNeurons.size()
    );

    for (const auto& neuron : inputNeurons) {
        thresholds.push_back(neuron->getThreshold());
    }

    for (const auto& neuron : layer1Neurons) {
        thresholds.push_back(neuron->getThreshold());
    }

    for (const auto& neuron : layer2Neurons) {
        thresholds.push_back(neuron->getThreshold());
    }

    for (const auto& neuron : layer3Neurons) {
        thresholds.push_back(neuron->getThreshold());
    }

    for (const auto& neuron : outputNeurons) {
        thresholds.push_back(neuron->getThreshold());
    }

    return thresholds;
}


void Brain::setNeuronThresholds(
    const std::vector<double>& thresholds
) {

    const std::size_t expected =
        inputNeurons.size() +
        layer1Neurons.size() +
        layer2Neurons.size() +
        layer3Neurons.size() +
        outputNeurons.size();

    if (thresholds.size() != expected) {
        throw std::invalid_argument(
            "Neuron threshold count does not match brain topology"
        );
    }

    std::size_t index = 0;

    for (auto& neuron : inputNeurons) {
        neuron->setThreshold(thresholds[index++]);
    }

    for (auto& neuron : layer1Neurons) {
        neuron->setThreshold(thresholds[index++]);
    }

    for (auto& neuron : layer2Neurons) {
        neuron->setThreshold(thresholds[index++]);
    }

    for (auto& neuron : layer3Neurons) {
        neuron->setThreshold(thresholds[index++]);
    }

    for (auto& neuron : outputNeurons) {
        neuron->setThreshold(thresholds[index++]);
    }
}


// ============================================================
// Synapse serialization
// ============================================================

std::vector<double> Brain::getSynapseStrengths() const {

    std::vector<double> strengths;

    strengths.reserve(synapses.size());

    for (const auto& synapse : synapses) {
        strengths.push_back(synapse->getStrength());
    }

    return strengths;
}


void Brain::setSynapseStrengths(
    const std::vector<double>& strengths
) {

    if (strengths.size() != synapses.size()) {
        throw std::invalid_argument(
            "Synapse strength count does not match brain topology"
        );
    }

    for (std::size_t i = 0; i < synapses.size(); ++i) {
        synapses[i]->setStrength(strengths[i]);
    }
}
