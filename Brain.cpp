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


void Brain::connect(
    Neuron* from,
    Neuron* to
) {

    auto synapse =
        std::make_unique<Synapse>(
            from,
            to
        );

    Synapse* synapsePtr =
        synapse.get();

    from->addSynapseOut(
        synapsePtr
    );

    to->addSynapseIn(
        synapsePtr
    );

    synapses.push_back(
        std::move(synapse)
    );
}
