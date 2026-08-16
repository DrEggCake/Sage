#include <iostream>

#include "Brain.h"

int main() {

    Brain brain(
        10,  // input neurons
        20,  // layer 1
        20,  // layer 2
        20,  // layer 3
        5,   // output neurons
        3,   // input -> layer 1
        3,   // layer 1 -> layer 2
        3,   // layer 2 -> layer 3
        3    // layer 3 -> output
    );

    std::cout << "Brain created\n";

    return 0;
}
