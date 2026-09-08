#ifndef SYNAPSE_H
#define SYNAPSE_H

class Neuron;

class Synapse {

public:

    Neuron* from;
    Neuron* to;

private:

    static constexpr double ELIGIBILITY_DECAY = 0.9;
    static constexpr double LAST_ACTIVATED_DECAY = 0.95;
    static constexpr double RESTING_STRENGTH = 0.05;
    static constexpr double LEAK_RATE = 0.005;
    static constexpr double PING_BOOST = 0.001;
    static constexpr double HEBB_BOOST = 0.05;
    static constexpr double DOPAMINE_GAIN = 0.02;
    static constexpr double MAX_STRENGTH = 2.0;

    double strength = 0.0;
    double actionPotential = 1.0;
    double eligibility = 0.0;
    double lastActivated = 0.0;

public:

    Synapse(Neuron* from, Neuron* to);

    void stimulate();
    void update();
    void applyReward(double amount);
    void applyDopamine(double amount);
    void decayEligibility(double factor);

    double getStrength() const;
    void setStrength(double strength);

    double getEligibility() const;
    void setEligibility(double eligibility);

    double getLastActivated() const;
    bool isPathActive() const;

    void reset();
};

#endif
