#include "Synapse.h"
#include "Neuron.h"

#include <cmath>

Synapse::Synapse(Neuron* from, Neuron* to) {
    this->from = from;
    this->to = to;
}

void Synapse::stimulate() {

    double signal =
        ACTION_POTENTIAL * STRENGTH;

    to->stimulate(signal);

    if (ELIGIBILITY < 1.0) {

        ELIGIBILITY +=
            std::abs(signal) *
            (1.0 - ELIGIBILITY);
    }
}

void Synapse::decayEligibility(double factor) {
    ELIGIBILITY *= factor;
}

double Synapse::getStrength() const {
    return STRENGTH;
}

void Synapse::setStrength(double strength) {
    STRENGTH = strength;
}

double Synapse::getEligibility() const {
    return ELIGIBILITY;
}

void Synapse::setEligibility(double eligibility) {
    ELIGIBILITY = eligibility;
}
