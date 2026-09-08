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

    static constexpr double VOLTAGE_LEAK = 0.85;
    static constexpr int REFRACTORY_TICKS = 1;
    static constexpr double THRESHOLD_ADJUST_RATE = 0.002;
    static constexpr double THRESHOLD_MIN = 0.3;
    static constexpr double THRESHOLD_MAX = 2.0;

    double voltage = 0.0;
    double threshold = 1.0;
    double eligibility = 0.0;
    double accumulatedDrive = 0.0;

    bool fired = false;
    int refractory = 0;
    int fireCount = 0;

    std::vector<Synapse*> synapsesIn;
    std::vector<Synapse*> synapsesOut;

    NeuronType neuronType;

public:

    Neuron(NeuronType neuronType);

    void update();
    void adjustThreshold(int targetRate);
    void resetFireCount();
    void reset();
    void setVoltage(double value);
    void pulse(double amount);
    void markFired();

    void addSynapseIn(Synapse* synapse);
    void addSynapseOut(Synapse* synapse);

    double getVoltage() const;
    double getThreshold() const;
    void setThreshold(double threshold);
    double getAccumulatedDrive() const;
    int getFireCount() const;

    bool hasFired() const;

    std::vector<Synapse*>& getSynapsesOut();
    std::vector<Synapse*>& getSynapsesIn();

    NeuronType getType() const;
};

#endif
