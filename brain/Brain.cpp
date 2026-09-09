#include "Brain.h"

#include <algorithm>
#include <stdexcept>


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
        inputNeurons.push_back(
            std::make_unique<Neuron>(Neuron::NeuronType::INPUT)
        );
    }

    for (int i = 0; i < LAYER_1_NEURONS; i++) {
        layer1Neurons.push_back(
            std::make_unique<Neuron>(Neuron::NeuronType::INTERNAL)
        );
    }

    for (int i = 0; i < LAYER_2_NEURONS; i++) {
        layer2Neurons.push_back(
            std::make_unique<Neuron>(Neuron::NeuronType::INTERNAL)
        );
    }

    for (int i = 0; i < LAYER_3_NEURONS; i++) {
        layer3Neurons.push_back(
            std::make_unique<Neuron>(Neuron::NeuronType::INTERNAL)
        );
    }

    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        outputNeurons.push_back(
            std::make_unique<Neuron>(Neuron::NeuronType::OUTPUT)
        );
    }
}


void Brain::buildConnections() {

    connectSliding(inputNeurons, layer1Neurons, INPUT_TO_LAYER_1_LIMIT);
    connectSliding(layer1Neurons, layer2Neurons, LAYER_1_TO_LAYER_2_LIMIT);
    connectSliding(layer2Neurons, layer3Neurons, LAYER_2_TO_LAYER_3_LIMIT);
    connectSliding(layer3Neurons, outputNeurons, LAYER_3_TO_OUTPUT_LIMIT);

    allSynapses.clear();
    for (auto& s : synapses) {
        allSynapses.push_back(s.get());
    }
}


void Brain::connectSliding(
    const std::vector<std::unique_ptr<Neuron>>& fromLayer,
    const std::vector<std::unique_ptr<Neuron>>& toLayer,
    int limit
) {

    int N1 = fromLayer.size();
    int N2 = toLayer.size();
    int K = std::min(limit, N2);

    if (K <= 0 || N1 <= 0 || N2 <= 0) return;

    for (int i = 0; i < N1; i++) {
        int start =
            N1 <= 1
            ? 0
            : static_cast<int>(
                static_cast<long long>(i) * (N2 - K) / (N1 - 1)
            );

        for (int j = 0; j < K; j++) {
            connect(fromLayer[i].get(), toLayer[start + j].get());
        }
    }
}


void Brain::connect(Neuron* from, Neuron* to) {

    auto synapse = std::make_unique<Synapse>(from, to);
    Synapse* synapsePtr = synapse.get();

    from->addSynapseOut(synapsePtr);
    to->addSynapseIn(synapsePtr);

    synapses.push_back(std::move(synapse));
}


// ============================================================
// Core simulation loop (ported from Sage-Java Brain.tick)
// ============================================================

void Brain::getAllNeurons(std::vector<Neuron*>& out) {
    out.clear();
    for (auto& n : inputNeurons)  out.push_back(n.get());
    for (auto& n : layer1Neurons) out.push_back(n.get());
    for (auto& n : layer2Neurons) out.push_back(n.get());
    for (auto& n : layer3Neurons) out.push_back(n.get());
    for (auto& n : outputNeurons) out.push_back(n.get());
}

void Brain::getAllNeuronsConst(std::vector<const Neuron*>& out) const {
    out.clear();
    for (auto& n : inputNeurons)  out.push_back(n.get());
    for (auto& n : layer1Neurons) out.push_back(n.get());
    for (auto& n : layer2Neurons) out.push_back(n.get());
    for (auto& n : layer3Neurons) out.push_back(n.get());
    for (auto& n : outputNeurons) out.push_back(n.get());
}

void Brain::tick() {

    for (Synapse* s : allSynapses) {
        if (s->from->hasFired()) {
            s->stimulate();
        }
    }

    std::vector<Neuron*> neurons;
    getAllNeurons(neurons);
    for (Neuron* n : neurons) {
        n->update();
    }

    for (Synapse* s : allSynapses) {
        s->update();
    }

    if (dopamine > 0.01) {
        for (Synapse* s : allSynapses) {
            if (s->isPathActive()) {
                s->applyDopamine(dopamine);
            }
        }
        dopamine *= dopamineDecay;
    }

    totalSteps++;
}


void Brain::setInput(int index, double value) {
    if (index >= 0 && index < INPUT_NEURONS) {
        inputNeurons[index]->setVoltage(value);
    }
}


double Brain::getOutput(int index) const {
    if (index >= 0 && index < OUTPUT_NEURONS) {
        return outputNeurons[index]->getVoltage();
    }
    return 0.0;
}


void Brain::resetOutputs() {
    for (auto& n : outputNeurons) {
        n->reset();
    }
}


void Brain::reward(double amount) {
    totalReward += amount;
    for (Synapse* s : allSynapses) {
        s->applyReward(amount * learningRate);
    }
}


void Brain::releaseDopamine(double amount) {
    dopamine = amount;
}


void Brain::adjustThresholds(int targetRate) {
    std::vector<Neuron*> neurons;
    getAllNeurons(neurons);
    for (Neuron* n : neurons) {
        n->adjustThreshold(targetRate);
        n->resetFireCount();
    }
}


void Brain::reset() {
    std::vector<Neuron*> neurons;
    getAllNeurons(neurons);
    for (Neuron* n : neurons) {
        n->reset();
    }
    for (Synapse* s : allSynapses) {
        s->reset();
    }
    dopamine = 0.0;
    totalReward = 0.0;
}


// ============================================================
// Serialization / persistence
// ============================================================

std::vector<int> Brain::getLayerSizes() const {
    return {INPUT_NEURONS, LAYER_1_NEURONS, LAYER_2_NEURONS, LAYER_3_NEURONS, OUTPUT_NEURONS};
}

std::vector<int> Brain::getWiringLimits() const {
    return {INPUT_TO_LAYER_1_LIMIT, LAYER_1_TO_LAYER_2_LIMIT, LAYER_2_TO_LAYER_3_LIMIT, LAYER_3_TO_OUTPUT_LIMIT};
}


std::vector<int> Brain::getFiredFlags() {
    std::vector<Neuron*> neurons;
    getAllNeurons(neurons);
    std::vector<int> flags;
    flags.reserve(neurons.size());
    for (const Neuron* n : neurons) {
        flags.push_back(n->hasFired() ? 1 : 0);
    }
    return flags;
}


double Brain::getLearningRate() const { return learningRate; }
void Brain::setLearningRate(double value) { learningRate = value; }

int Brain::getEpisodesTrained() const { return episodesTrained; }
int Brain::getSuccesses() const { return successes; }
double Brain::getTotalReward() const { return totalReward; }
long Brain::getTotalSteps() const { return totalSteps; }
long Brain::getTotalFired() const { return totalFired; }
double Brain::getBestReward() const { return bestReward; }

void Brain::setEpisodesTrained(int value) { episodesTrained = value; }
void Brain::setSuccesses(int value) { successes = value; }
void Brain::setTotalReward(double value) { totalReward = value; }
void Brain::setTotalSteps(long value) { totalSteps = value; }
void Brain::setTotalFired(long value) { totalFired = value; }
void Brain::setBestReward(double value) { bestReward = value; }


// ============================================================
// Neuron serialization
// ============================================================

std::vector<double> Brain::getNeuronThresholds() const {

    std::vector<double> thresholds;
    thresholds.reserve(
        inputNeurons.size() + layer1Neurons.size() +
        layer2Neurons.size() + layer3Neurons.size() + outputNeurons.size()
    );

    for (const auto& n : inputNeurons)  thresholds.push_back(n->getThreshold());
    for (const auto& n : layer1Neurons) thresholds.push_back(n->getThreshold());
    for (const auto& n : layer2Neurons) thresholds.push_back(n->getThreshold());
    for (const auto& n : layer3Neurons) thresholds.push_back(n->getThreshold());
    for (const auto& n : outputNeurons) thresholds.push_back(n->getThreshold());

    return thresholds;
}


void Brain::setNeuronThresholds(const std::vector<double>& thresholds) {

    const std::size_t expected =
        inputNeurons.size() + layer1Neurons.size() +
        layer2Neurons.size() + layer3Neurons.size() + outputNeurons.size();

    if (thresholds.size() != expected) {
        throw std::invalid_argument("Neuron threshold count does not match brain topology");
    }

    std::size_t index = 0;
    for (auto& n : inputNeurons)  n->setThreshold(thresholds[index++]);
    for (auto& n : layer1Neurons) n->setThreshold(thresholds[index++]);
    for (auto& n : layer2Neurons) n->setThreshold(thresholds[index++]);
    for (auto& n : layer3Neurons) n->setThreshold(thresholds[index++]);
    for (auto& n : outputNeurons) n->setThreshold(thresholds[index++]);
}


// ============================================================
// Synapse serialization
// ============================================================

std::vector<double> Brain::getSynapseStrengths() const {
    std::vector<double> strengths;
    strengths.reserve(synapses.size());
    for (const auto& s : synapses) {
        strengths.push_back(s->getStrength());
    }
    return strengths;
}


void Brain::setSynapseStrengths(const std::vector<double>& strengths) {
    if (strengths.size() != synapses.size()) {
        throw std::invalid_argument("Synapse strength count does not match brain topology");
    }
    for (std::size_t i = 0; i < synapses.size(); ++i) {
        synapses[i]->setStrength(strengths[i]);
    }
}
