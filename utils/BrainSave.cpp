#include "BrainSave.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace BrainSave {

namespace {

constexpr const char* PREFIX = "brain_";
constexpr const char* SUFFIX = ".json";

using json = nlohmann::json;

struct BrainDto {

    std::vector<int> layerSizes;
    std::vector<int> wiringLimits;

    double learningRate;
    double eligibilityDecay;
    double leakRate;

    std::vector<double> thresholds;
    std::vector<double> synapseStrengths;

    int episodesTrained;
    int successes;

    double totalReward;

    long totalSteps;
    long totalFired;

    double bestReward;
};


void to_json(json& j, const BrainDto& dto) {

    j = json{
        {"layerSizes", dto.layerSizes},
        {"wiringLimits", dto.wiringLimits},

        {"learningRate", dto.learningRate},
        {"eligibilityDecay", dto.eligibilityDecay},
        {"leakRate", dto.leakRate},

        {"thresholds", dto.thresholds},
        {"synapseStrengths", dto.synapseStrengths},

        {"episodesTrained", dto.episodesTrained},
        {"successes", dto.successes},

        {"totalReward", dto.totalReward},

        {"totalSteps", dto.totalSteps},
        {"totalFired", dto.totalFired},

        {"bestReward", dto.bestReward}
    };
}


void from_json(const json& j, BrainDto& dto) {

    j.at("layerSizes").get_to(dto.layerSizes);
    j.at("wiringLimits").get_to(dto.wiringLimits);

    j.at("learningRate").get_to(dto.learningRate);
    j.at("eligibilityDecay").get_to(dto.eligibilityDecay);
    j.at("leakRate").get_to(dto.leakRate);

    j.at("thresholds").get_to(dto.thresholds);
    j.at("synapseStrengths").get_to(dto.synapseStrengths);

    j.at("episodesTrained").get_to(dto.episodesTrained);
    j.at("successes").get_to(dto.successes);

    j.at("totalReward").get_to(dto.totalReward);

    j.at("totalSteps").get_to(dto.totalSteps);
    j.at("totalFired").get_to(dto.totalFired);

    j.at("bestReward").get_to(dto.bestReward);
}


BrainDto toDto(const Brain& brain) {

    BrainDto dto;

    dto.layerSizes = brain.getLayerSizes();
    dto.wiringLimits = brain.getWiringLimits();

    dto.learningRate = brain.getLearningRate();
    dto.eligibilityDecay = brain.getEligibilityDecay();
    dto.leakRate = brain.getLeakRate();

    dto.thresholds = brain.getNeuronThresholds();
    dto.synapseStrengths = brain.getSynapseStrengths();

    dto.episodesTrained = brain.getEpisodesTrained();
    dto.successes = brain.getSuccesses();

    dto.totalReward = brain.getTotalReward();

    dto.totalSteps = brain.getTotalSteps();
    dto.totalFired = brain.getTotalFired();

    dto.bestReward = brain.getBestReward();

    return dto;
}


int nextNumber(const std::filesystem::path& dir) {

    if (!std::filesystem::exists(dir)) {
        return 1;
    }

    int maxNumber = 0;

    const std::regex pattern(R"(brain_(\d+)\.json)");

    for (const auto& entry :
         std::filesystem::directory_iterator(dir)) {

        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string filename =
            entry.path().filename().string();

        std::smatch match;

        if (std::regex_match(filename, match, pattern)) {

            const int number =
                std::stoi(match[1].str());

            maxNumber =
                std::max(maxNumber, number);
        }
    }

    return maxNumber + 1;
}

} // anonymous namespace


std::filesystem::path save(
    const Brain& brain,
    const std::filesystem::path& dir
) {

    if (!std::filesystem::exists(dir)) {

        if (!std::filesystem::create_directories(dir)) {
            throw std::runtime_error(
                "Could not create directory: " +
                dir.string()
            );
        }
    }

    const int number = nextNumber(dir);

    const auto file =
        dir /
        (std::string(PREFIX) +
         std::to_string(number) +
         SUFFIX);

    std::ofstream out(file);

    if (!out) {
        throw std::runtime_error(
            "Could not open file for writing: " +
            file.string()
        );
    }

    const BrainDto dto = toDto(brain);

    json j = dto;

    // Pretty-print JSON, equivalent to Gson's
    // GsonBuilder().setPrettyPrinting()
    out << j.dump(4) << '\n';

    if (!out) {
        throw std::runtime_error(
            "Failed while writing: " +
            file.string()
        );
    }

    return file;
}


Brain load(const std::filesystem::path& file) {

    std::ifstream in(file);

    if (!in) {
        throw std::runtime_error(
            "Could not open brain file: " +
            file.string()
        );
    }

    json j;

    try {
        in >> j;
    }
    catch (const json::parse_error& e) {

        throw std::runtime_error(
            "Invalid JSON in brain file: " +
            file.string() +
            ": " +
            e.what()
        );
    }

    BrainDto dto;

    try {
        dto = j.get<BrainDto>();
    }
    catch (const json::exception& e) {

        throw std::runtime_error(
            "Not a valid Sage brain file: " +
            file.string() +
            ": " +
            e.what()
        );
    }

    if (dto.layerSizes.size() != 5 ||
        dto.wiringLimits.size() != 4) {

        throw std::runtime_error(
            "Not a Sage brain file: " +
            file.string()
        );
    }


    // Rebuild the deterministic topology.
    Brain brain(
        dto.layerSizes[0],
        dto.layerSizes[1],
        dto.layerSizes[2],
        dto.layerSizes[3],
        dto.layerSizes[4],

        dto.wiringLimits[0],
        dto.wiringLimits[1],
        dto.wiringLimits[2],
        dto.wiringLimits[3]
    );


    // Restore learning configuration.
    brain.setLearningRate(
        dto.learningRate
    );

    brain.setEligibilityDecay(
        dto.eligibilityDecay
    );

    brain.setLeakRate(
        dto.leakRate
    );


    // Restore trained neuron state.
    if (!dto.thresholds.empty()) {

        brain.setNeuronThresholds(
            dto.thresholds
        );
    }


    // Restore trained synapse state.
    if (!dto.synapseStrengths.empty()) {

        brain.setSynapseStrengths(
            dto.synapseStrengths
        );
    }


    // Restore training statistics.
    brain.setEpisodesTrained(
        dto.episodesTrained
    );

    brain.setSuccesses(
        dto.successes
    );

    brain.setTotalReward(
        dto.totalReward
    );

    brain.setTotalSteps(
        dto.totalSteps
    );

    brain.setTotalFired(
        dto.totalFired
    );

    brain.setBestReward(
        dto.bestReward
    );

    return brain;
}


int nextBrainNumber(
    const std::filesystem::path& dir
) {
    return nextNumber(dir);
}

} // namespace BrainSave
