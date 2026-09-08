#include "Synapse.h"
#include "Neuron.h"

#include <algorithm>
#include <cmath>

Synapse::Synapse(Neuron* from, Neuron* to) {
    this->from = from;
    this->to = to;
}

void Synapse::stimulate() {

    double signal = actionPotential * strength;

    to->pulse(signal);

    eligibility = 1.0;
    lastActivated = 1.0;
}

void Synapse::update() {

    eligibility *= ELIGIBILITY_DECAY;
    lastActivated *= LAST_ACTIVATED_DECAY;

    if (std::abs(strength - RESTING_STRENGTH) > 0.001) {
        if (strength > RESTING_STRENGTH) {
            strength -= LEAK_RATE;
            if (strength < RESTING_STRENGTH) strength = RESTING_STRENGTH;
        } else {
            strength += LEAK_RATE;
            if (strength > RESTING_STRENGTH) strength = RESTING_STRENGTH;
        }
    }

    if (eligibility > 0.05 && to->hasFired()) {
        strength += HEBB_BOOST;
    }

    if (from->hasFired()) {
        strength += PING_BOOST;
    }

    strength = std::clamp(strength, -MAX_STRENGTH, MAX_STRENGTH);
}

void Synapse::applyReward(double amount) {
    strength += amount * eligibility;
    strength = std::clamp(strength, -MAX_STRENGTH, MAX_STRENGTH);
}

void Synapse::applyDopamine(double amount) {
    strength += amount * lastActivated * DOPAMINE_GAIN;
    strength = std::clamp(strength, -MAX_STRENGTH, MAX_STRENGTH);
}

void Synapse::decayEligibility(double factor) {
    eligibility *= factor;
}

double Synapse::getStrength() const {
    return strength;
}

void Synapse::setStrength(double strength) {
    this->strength = strength;
}

double Synapse::getEligibility() const {
    return eligibility;
}

void Synapse::setEligibility(double eligibility) {
    this->eligibility = eligibility;
}

double Synapse::getLastActivated() const {
    return lastActivated;
}

bool Synapse::isPathActive() const {
    return lastActivated > 0.01;
}

void Synapse::reset() {
    eligibility = 0.0;
    lastActivated = 0.0;
}
