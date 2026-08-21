#include "Neuron.h"
#include "Synapse.h"

Neuron::Neuron(NeuronType neuronType) {
    this->neuronType = neuronType;
}

void Neuron::stimulate(double amount) {
    VOLTAGE += amount;
}

void Neuron::fire() {
    for (Synapse* s : synapsesOut) {
        s->stimulate();
    }
}

void Neuron::spike() {

    fire();

    VOLTAGE -= THRESHOLD;

    if (VOLTAGE < 0.0) {
        VOLTAGE = 0.0;
    }

    firedThisEpisode = true;
    ELIGIBILITY = 1.0;
}

void Neuron::markFired() {

    VOLTAGE = 0.0;
    firedThisEpisode = true;
    ELIGIBILITY = 1.0;
}

void Neuron::resetVoltage() {
    VOLTAGE = 0.0;
}

void Neuron::reset() {

    VOLTAGE = 0.0;
    ELIGIBILITY = 0.0;
    firedThisEpisode = false;
}

void Neuron::addSynapseIn(Synapse* synapse) {
    synapsesIn.push_back(synapse);
}

void Neuron::addSynapseOut(Synapse* synapse) {
    synapsesOut.push_back(synapse);
}

double Neuron::getVoltage() const {
    return VOLTAGE;
}

double Neuron::getThreshold() const {
    return THRESHOLD;
}

void Neuron::setThreshold(double threshold) {
    THRESHOLD = threshold;
}

bool Neuron::firedThisEpisodeCheck() const {
    return firedThisEpisode;
}

std::vector<Synapse*>& Neuron::getSynapsesOut() {
    return synapsesOut;
}

Neuron::NeuronType Neuron::getType() const {
    return neuronType;
}
