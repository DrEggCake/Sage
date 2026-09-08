#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <filesystem>
#include <iomanip>

#include <mujoco/mujoco.h>

#include "brain/Brain.h"
#include "utils/BrainSave.h"

// SNN-driven MuJoCo task: the brain controls the 2-DOF arm to reach a fixed
// target. Each physics step feeds sensors (ee pos, ee-target error, joint
// angles) into the brain, ticks it, and maps the 4 outputs onto the shoulder
// and elbow motor ctrls. Records the end-effector trajectory (CSV) plus a
// per-tick neuron firing log, and saves the trained brain.
//   Usage: snn_drive [--episodes N] [--steps N] [--epsilon-start E]
//                    [--epsilon-end E] [--seed S] [--brain FILE]
//                    [--fire-log FILE] [--record FILE] [--save-dir DIR]
//                    [--model arm_reach]
//   --record FILE   write ee trajectory (time,x,y,z,vx,vy,vz) for the replay player
//   --fire-log FILE write SNN firing log CSV (with .meta.json sidecar)

namespace {

double clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int name2id(mjModel* m, int type, const std::string& name) {
    int id = mj_name2id(m, type, name.c_str());
    if (id < 0) {
        std::cerr << "Failed to find " << name << "\n";
    }
    return id;
}

std::string findModel(const std::string& name) {
    std::filesystem::path p = std::filesystem::current_path();
    while (p.parent_path() != p) {
        auto candidate = p / "tests" / "models" / (name + ".xml");
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
        p = p.parent_path();
    }
    return (std::filesystem::current_path() / "tests" / "models" / (name + ".xml")).string();
}

} // namespace

int main(int argc, char** argv) {

    int episodes = 120;
    int steps = 240;
    double epsilonStart = 1.0;
    double epsilonEnd = 0.05;
    double learningRate = 0.02;
    unsigned seed = 42;
    double shaping = 1.0;
    std::string brainFile;
    std::string fireLog;
    std::string recordCsv;
    std::string saveDir = "data/brain";
    std::string modelName = "arm_reach";
    const double reachRadius = 0.15;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto has = i + 1 < argc;
        auto num = [&](const std::string& s) {
            try { return std::stod(s); }
            catch (...) { return 0.0; }
        };
        if (a == "--episodes" && has) episodes = (int)num(argv[++i]);
        else if (a == "--steps" && has) steps = (int)num(argv[++i]);
        else if (a == "--epsilon-start" && has) epsilonStart = num(argv[++i]);
        else if (a == "--epsilon-end" && has) epsilonEnd = num(argv[++i]);
        else if (a == "--learning-rate" && has) learningRate = num(argv[++i]);
        else if (a == "--seed" && has) seed = (unsigned)num(argv[++i]);
        else if (a == "--shaping" && has) shaping = num(argv[++i]);
        else if (a == "--brain" && has) brainFile = argv[++i];
        else if (a == "--fire-log" && has) fireLog = argv[++i];
        else if (a == "--record" && has) recordCsv = argv[++i];
        else if (a == "--save-dir" && has) saveDir = argv[++i];
        else if (a == "--model" && has) modelName = argv[++i];
        else if (a == "--help") {
            std::cout << "Usage: snn_drive [--episodes N] [--steps N] [--epsilon-start E]\n"
                      << "                 [--epsilon-end E] [--learning-rate R] [--seed S]\n"
                      << "                 [--shaping W] [--brain FILE] [--fire-log FILE]\n"
                      << "                 [--record FILE] [--save-dir DIR] [--model arm_reach]\n";
            return 0;
        }
    }

    std::string modelPath = findModel(modelName);
    char error[1000];
    mjModel* m = mj_loadXML(modelPath.c_str(), nullptr, error, 1000);
    if (!m) {
        std::cerr << "Failed to load model: " << error << "\n";
        return 1;
    }
    mjData* d = mj_makeData(m);

    const int eeBody = name2id(m, mjOBJ_BODY, "ee");
    const int senEe = name2id(m, mjOBJ_SENSOR, "ee_pos");
    const int senTgt = name2id(m, mjOBJ_SENSOR, "target_pos");
    const int senSh = name2id(m, mjOBJ_SENSOR, "shoulder_pos");
    const int senEl = name2id(m, mjOBJ_SENSOR, "elbow_pos");
    const int actSh = name2id(m, mjOBJ_ACTUATOR, "shoulder_motor");
    const int actEl = name2id(m, mjOBJ_ACTUATOR, "elbow_motor");
    if (eeBody < 0 || senEe < 0 || senTgt < 0 || senSh < 0 || senEl < 0 || actSh < 0 || actEl < 0) {
        return 1;
    }

    const int adrEe = m->sensor_adr[senEe];
    const int adrTgt = m->sensor_adr[senTgt];
    const int adrSh = m->sensor_adr[senSh];
    const int adrEl = m->sensor_adr[senEl];

    Brain brain(8, 16, 16, 16, 4, 3, 3, 3, 3);
    if (!brainFile.empty()) {
        try {
            brain = BrainSave::load(brainFile);
            std::cout << "Loaded brain " << brainFile << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Brain load failed: " << e.what() << " (training fresh)\n";
        }
    }
    brain.setLearningRate(learningRate);

    std::cout << "SNN-driven arm reach (" << modelName << ")\n"
              << "  episodes " << episodes << ", steps/ep " << steps
              << " (" << steps * m->opt.timestep << " s), lr " << learningRate
              << ", seed " << seed << "\n"
              << "  epsilon " << epsilonStart << " -> " << epsilonEnd << "\n";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    std::ofstream csvOut;
    if (!recordCsv.empty()) {
        std::filesystem::create_directories(std::filesystem::path(recordCsv).parent_path());
        csvOut.open(recordCsv);
        csvOut << "time,x,y,z,vx,vy,vz\n";
    }

    std::ofstream logger;
    if (!fireLog.empty()) {
        auto lp = std::filesystem::path(fireLog);
        if (lp.has_parent_path()) std::filesystem::create_directories(lp.parent_path());
        logger.open(fireLog);
        logger << "# topology 8,16,16,16,4\n";
        logger << "# task arm_reach episodes " << episodes << " steps " << steps
               << " timestep " << m->opt.timestep << " hz " << (1.0 / m->opt.timestep)
               << " seed " << seed << " lr " << learningRate
               << " epsilonStart " << epsilonStart << " epsilonEnd " << epsilonEnd << "\n";
        logger << "ep,tick";
        for (int i = 0; i < 8; i++) logger << ",i" << i;
        for (int i = 0; i < 60; i++) logger << ",f" << i;
        logger << '\n';
    }

    double totalRewardAcc = 0.0;
    int reached = 0;
    int reportEvery = std::max(1, episodes / 5);

    for (int ep = 0; ep < episodes; ep++) {

        double eps = epsilonEnd;
        if (episodes > 1) {
            eps = epsilonStart + (epsilonEnd - epsilonStart) *
                                 ((double)ep / (double)(episodes - 1));
        }

        mj_resetData(m, d);
        brain.reset();
        brain.resetOutputs();

        double minDist = 1e9;
        double lastX = 0, lastY = 0, lastZ = 0;
        bool haveLast = false;

        for (int t = 0; t < steps; t++) {

            const double* ee = d->sensordata + adrEe;
            const double* tg = d->sensordata + adrTgt;
            const double sh = d->sensordata[adrSh];
            const double el = d->sensordata[adrEl];

            const double dist = std::sqrt(
                std::pow(ee[0] - tg[0], 2) + std::pow(ee[1] - tg[1], 2) + std::pow(ee[2] - tg[2], 2));
            if (dist < minDist) minDist = dist;

            std::vector<double> ins(8);
            ins[0] = clamp((ee[0] - tg[0]) / 0.35, -2.0, 2.0);
            ins[1] = clamp((ee[1] - tg[1]) / 0.35, -2.0, 2.0);
            ins[2] = clamp((ee[2] - tg[2]) / 0.35, -2.0, 2.0);
            ins[3] = clamp(ee[0] / 0.8, -2.0, 2.0);
            ins[4] = clamp(ee[1] / 0.8, -2.0, 2.0);
            ins[5] = clamp(ee[2] / 0.8, -2.0, 2.0);
            ins[6] = clamp(sh / 1.5708, -2.0, 2.0);
            ins[7] = clamp(el / 1.5708, -2.0, 2.0);

            for (int i = 0; i < 8; i++) {
                brain.setInput(i, ins[i]);
            }

            brain.tick();

            if (shaping > 0.0) {
                brain.reward(0.05 * shaping * clamp(0.2 - 2.0 * dist, -0.2, 0.2));
            }

            const bool explore = coin(rng) < eps;
            const double cSh = explore ? unit(rng) : clamp(0.45 + 2.0 * brain.getOutput(0), -1.0, 1.0);
            const double cEl = explore ? unit(rng) : clamp(0.15 + 2.0 * brain.getOutput(1), -1.0, 1.0);
            d->ctrl[actSh] = cSh;
            d->ctrl[actEl] = cEl;

            if (logger.is_open()) {
                logger << ep << ',' << t;
                for (double v : ins) logger << ',' << v;
                for (int f : brain.getFiredFlags()) logger << ',' << f;
                logger << '\n';
            }

            mj_step(m, d);

            if (csvOut.is_open()) {
                const double* p = d->xpos + 3 * eeBody;
                double vx = 0, vy = 0, vz = 0;
                if (haveLast) {
                    const double iv = 1.0 / m->opt.timestep;
                    vx = (p[0] - lastX) * iv; vy = (p[1] - lastY) * iv; vz = (p[2] - lastZ) * iv;
                }
                lastX = p[0]; lastY = p[1]; lastZ = p[2]; haveLast = true;
                csvOut << std::fixed << std::setprecision(6)
                       << d->time << ',' << p[0] << ',' << p[1] << ',' << p[2] << ','
                       << vx << ',' << vy << ',' << vz << '\n';
            }
        }

        const double* eeEnd = d->sensordata + adrEe;
        const double* tgEnd = d->sensordata + adrTgt;
        const double endDist = std::sqrt(
            std::pow(eeEnd[0] - tgEnd[0], 2) + std::pow(eeEnd[1] - tgEnd[1], 2) + std::pow(eeEnd[2] - tgEnd[2], 2));
        const bool got = endDist <= reachRadius;
        const double reward = got ? 1.0 : -0.5;
        brain.reward(reward);
        totalRewardAcc += reward;
        if (got) reached++;
        brain.adjustThresholds(2);

        if ((ep == 0) || ((ep + 1) % reportEvery == 0)) {
            std::cout << "  ep " << (ep + 1) << "/" << episodes
                      << "  epsilon " << round(eps * 1000) / 1000
                      << "  reached " << reached << "/" << (ep + 1)
                      << "  reward+ " << round(totalRewardAcc * 1000) / 1000
                      << "  endDist " << round(endDist * 1000) / 1000
                      << "  minDist " << round(minDist * 1000) / 1000 << " m\n";
        }
    }

    double rate = (double)reached / (double)episodes * 100.0;
    std::cout << "Done: " << episodes << " eps, reached " << reached
              << " (" << round(rate * 10) / 10 << "%), total reward "
              << round(totalRewardAcc * 1000) / 1000 << ", seed " << seed << "\n";

    brain.setEpisodesTrained(episodes);
    brain.setSuccesses(reached);
    brain.setTotalReward(totalRewardAcc);
    brain.setBestReward(reached > 0 ? 1.0 : -0.5);
    brain.setTotalSteps((long)episodes * steps);

    std::filesystem::create_directories(saveDir);
    auto file = BrainSave::save(brain, saveDir);
    std::cout << "Saved brain to " << file << "\n";

    if (!fireLog.empty()) {
        std::string metaPath = fireLog + ".meta.json";
        std::ofstream meta(metaPath);
        meta << "{\n"
             << "  \"brain\": \"" << file.filename().string() << "\",\n"
             << "  \"layerSizes\": [8,16,16,16,4],\n"
             << "  \"wiringLimits\": [3,3,3,3],\n"
             << "  \"episodes\": " << episodes << ",\n"
             << "  \"decisionTicks\": " << steps << ",\n"
             << "  \"stepsPerEpisode\": " << steps << ",\n"
             << "  \"hz\": " << (1.0 / m->opt.timestep) << ",\n"
             << "  \"task\": \"arm_reach\",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"learningRate\": " << learningRate << ",\n"
             << "  \"epsilonStart\": " << epsilonStart << ",\n"
             << "  \"epsilonEnd\": " << epsilonEnd << ",\n"
             << "  \"successes\": " << reached << ",\n"
             << "  \"reachedRate\": " << rate << ",\n"
             << "  \"totalReward\": " << totalRewardAcc << ",\n"
             << "  \"episodesTrained\": " << episodes << "\n"
             << "}\n";
        std::cout << "Fire log: " << fireLog << " (+ sidecar " << metaPath << ")\n";
    }

    if (csvOut.is_open()) {
        std::cout << "Recorded " << episodes << " x " << steps << " ee samples to " << recordCsv << "\n";
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}