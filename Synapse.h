#ifndef SYNAPSE_H
#define SYNAPSE_H

class Neuron;

class Synapse {

public:

    Neuron* from;
    Neuron* to;

private:

    double STRENGTH = 0.0;
    double ACTION_POTENTIAL = 1.0;
    double ELIGIBILITY = 0.0;

public:

    Synapse(Neuron* from, Neuron* to);

    void stimulate();
    void decayEligibility(double factor);

    double getStrength();
    void setStrength(double strength);

    double getEligibility();
    void setEligibility(double eligibility);
};

#endif
