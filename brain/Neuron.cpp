#include "Neuron.h"
#include "Synapse.h"

#include <algorithm>
#include <cmath>

Neuron::Neuron(NeuronType neuronType) {
    this->neuronType = neuronType;
}

void Neuron::update() {

    if (refractory > 0) {
        refractory--;
        fired = false;
        return;
    }

    if (voltage >= threshold) {
        fired = true;
        fireCount++;
        accumulatedDrive += voltage;
        voltage -= threshold;
        if (voltage < 0.0) voltage = 0.0;
        for (Synapse* s : synapsesOut) {
            s->stimulate();
        }
        refractory = REFRACTORY_TICKS;
    } else {
        fired = false;
        voltage *= VOLTAGE_LEAK;
    }
}

void Neuron::adjustThreshold(int targetRate) {
    if (neuronType == NeuronType::INPUT || neuronType == NeuronType::OUTPUT) {
        return;
    }
    if (fireCount > targetRate) {
        threshold += THRESHOLD_ADJUST_RATE;
    } else if (fireCount < targetRate) {
        threshold -= THRESHOLD_ADJUST_RATE;
    }
    threshold = std::clamp(threshold, THRESHOLD_MIN, THRESHOLD_MAX);
}

void Neuron::resetFireCount() {
    fireCount = 0;
}

void Neuron::reset() {
    voltage = 0.0;
    eligibility = 0.0;
    accumulatedDrive = 0.0;
    fired = false;
    refractory = 0;
}

void Neuron::setVoltage(double value) {
    voltage = value;
}

void Neuron::pulse(double amount) {
    voltage += amount;
}

void Neuron::markFired() {
    fired = true;
    fireCount++;
}

void Neuron::addSynapseIn(Synapse* synapse) {
    synapsesIn.push_back(synapse);
}

void Neuron::addSynapseOut(Synapse* synapse) {
    synapsesOut.push_back(synapse);
}

double Neuron::getVoltage() const {
    return voltage;
}

double Neuron::getThreshold() const {
    return threshold;
}

void Neuron::setThreshold(double threshold) {
    this->threshold = threshold;
}

double Neuron::getAccumulatedDrive() const {
    return accumulatedDrive;
}

int Neuron::getFireCount() const {
    return fireCount;
}

bool Neuron::hasFired() const {
    return fired;
}

std::vector<Synapse*>& Neuron::getSynapsesOut() {
    return synapsesOut;
}

std::vector<Synapse*>& Neuron::getSynapsesIn() {
    return synapsesIn;
}

Neuron::NeuronType Neuron::getType() const {
    return neuronType;
}
