#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Synapse;

class Neuron {

public:

    enum class NeuronType {
        INPUT,
        INTERNAL,
        OUTPUT
    };

private:

    double VOLTAGE = 0.0;
    double THRESHOLD = 1.0;
    double ELIGIBILITY = 0.0;

    bool firedThisEpisode = false;

    std::vector<Synapse*> synapsesIn;
    std::vector<Synapse*> synapsesOut;

    NeuronType neuronType;

public:

    Neuron(NeuronType neuronType);

    void stimulate(double amount);
    void fire();
    void spike();
    void markFired();
    void resetVoltage();
    void reset();

    void addSynapseIn(Synapse* synapse);
    void addSynapseOut(Synapse* synapse);

    double getVoltage();
    double getThreshold();
    void setThreshold(double threshold);

    bool firedThisEpisodeCheck();

    std::vector<Synapse*>& getSynapsesOut();

    NeuronType getType();
};

#endif
