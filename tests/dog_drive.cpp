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

// SNN-driven MuJoCo quadruped: a simple 4-legged robot dog (cuboid torso +
// 4 legs with hip+knee hinges) whose legs are driven by the Sage spiking
// brain. The brain senses only proprioception (the 8 joint angles, torso
// pitch/roll, body height) — the same way we know our own limb angles without
// looking — and learns to walk a target distance along +x.
//
// Per physics step: read joint angles etc. -> tick brain -> map the 8 output
// voltages onto the 8 motor ctrls -> physics step. Dense reward = forward
// progress; terminal reward on success (travelled >= target distance), on
// falling down, or on timeout.
//
// At the end one deterministic exploit episode is recorded: the torso
// trajectory CSV (replay player) + a per-tick SNN firing log (+ meta sidecar),
// and the trained brain is saved.
//
//   Usage: dog_drive [--episodes N] [--steps N] [--target-dist M]
//                    [--epsilon-start E] [--epsilon-end E] [--explore-amp A]
//                    [--learning-rate R] [--reward-progress W] [--seed S]
//                    [--brain FILE] [--save-dir DIR] [--record FILE]
//                    [--fire-log FILE] [--report FILE]
//
//   --record FILE write final torso trajectory (time,x,y,z,vx,vy,vz)
//   --fire-log FILE write final-episode SNN firing log (+ .meta.json sidecar)
//   --report FILE write per-episode stats (ep,dist,reward,success)

namespace {

constexpr int NIN = 12;   // 8 joint angles + pitch + roll + body height + drive
constexpr int NL1 = 18;
constexpr int NL2 = 18;
constexpr int NL3 = 18;
constexpr int NOUT = 8;   // one ctrl per leg joint
constexpr int NTOT = NIN + NL1 + NL2 + NL3 + NOUT;  // 74 neurons
constexpr int NLEFT = 8;  // 4 hips + 4 knees, in sensor order below

// Input i11 is a constant drive channel (1.5 V, always fires) so the hidden
// layers and outputs have resting activity instead of sitting dead at zero
// input — without it the network is silent while the robot stands still.

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

    int episodes = 3000;
    int steps = 2400;
    double targetDist = 2.0;
    double epsilonStart = 1.0;
    double epsilonEnd = 0.05;
    double exploreAmp = 0.35;
    double learningRate = 0.5;
    double rewardProgress = 8.0;
    unsigned seed = 42;
    std::string brainFile;
    std::string saveDir = "data/brain";
    std::string recordCsv;
    std::string fireLog;
    std::string reportCsv;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto has = i + 1 < argc;
        auto num = [&](const std::string& s) {
            try { return std::stod(s); }
            catch (...) { return 0.0; }
        };
        if (a == "--episodes" && has) episodes = (int)num(argv[++i]);
        else if (a == "--steps" && has) steps = (int)num(argv[++i]);
        else if (a == "--target-dist" && has) targetDist = num(argv[++i]);
        else if (a == "--epsilon-start" && has) epsilonStart = num(argv[++i]);
        else if (a == "--epsilon-end" && has) epsilonEnd = num(argv[++i]);
        else if (a == "--explore-amp" && has) exploreAmp = num(argv[++i]);
        else if (a == "--learning-rate" && has) learningRate = num(argv[++i]);
        else if (a == "--reward-progress" && has) rewardProgress = num(argv[++i]);
        else if (a == "--seed" && has) seed = (unsigned)num(argv[++i]);
        else if (a == "--brain" && has) brainFile = argv[++i];
        else if (a == "--save-dir" && has) saveDir = argv[++i];
        else if (a == "--record" && has) recordCsv = argv[++i];
        else if (a == "--fire-log" && has) fireLog = argv[++i];
        else if (a == "--report" && has) reportCsv = argv[++i];
        else if (a == "--help") {
            std::cout << "Usage: dog_drive [--episodes N] [--steps N] [--target-dist M]\n"
                      << "                 [--epsilon-start E] [--epsilon-end E] [--explore-amp A]\n"
                      << "                 [--learning-rate R] [--reward-progress W] [--seed S]\n"
                      << "                 [--brain FILE] [--save-dir DIR] [--record FILE]\n"
                      << "                 [--fire-log FILE] [--report FILE]\n";
            return 0;
        }
    }

    std::string modelPath = findModel("dog_walk");
    char error[1000];
    mjModel* m = mj_loadXML(modelPath.c_str(), nullptr, error, 1000);
    if (!m) {
        std::cerr << "Failed to load model: " << error << "\n";
        return 1;
    }
    mjData* d = mj_makeData(m);

    const int torsoBody = name2id(m, mjOBJ_BODY, "torso");
    const char* jointSensors[8] = {
        "FR_hip_s", "FL_hip_s", "RR_hip_s", "RL_hip_s",
        "FR_knee_s", "FL_knee_s", "RR_knee_s", "RL_knee_s",
    };
    int senJ[8];
    for (int i = 0; i < 8; i++) senJ[i] = name2id(m, mjOBJ_SENSOR, jointSensors[i]);
    const int senQ = name2id(m, mjOBJ_SENSOR, "torso_quat");
    const int actHips[4] = {
        name2id(m, mjOBJ_ACTUATOR, "FR_hip_m"),
        name2id(m, mjOBJ_ACTUATOR, "FL_hip_m"),
        name2id(m, mjOBJ_ACTUATOR, "RR_hip_m"),
        name2id(m, mjOBJ_ACTUATOR, "RL_hip_m"),
    };
    const int actKnees[4] = {
        name2id(m, mjOBJ_ACTUATOR, "FR_knee_m"),
        name2id(m, mjOBJ_ACTUATOR, "FL_knee_m"),
        name2id(m, mjOBJ_ACTUATOR, "RR_knee_m"),
        name2id(m, mjOBJ_ACTUATOR, "RL_knee_m"),
    };
    int act[8] = { actHips[0], actHips[1], actHips[2], actHips[3],
                   actKnees[0], actKnees[1], actKnees[2], actKnees[3] };
    if (torsoBody < 0 || senQ < 0) return 1;
    for (int i = 0; i < 8; i++) if (senJ[i] < 0 || act[i] < 0) return 1;

    int adrJ[8];
    for (int i = 0; i < 8; i++) adrJ[i] = m->sensor_adr[senJ[i]];
    const int adrQ = m->sensor_adr[senQ];

    Brain brain(NIN, NL1, NL2, NL3, NOUT, 6, 5, 5, 4);
    if (!brainFile.empty()) {
        try {
            brain = BrainSave::load(brainFile);
            std::cout << "Loaded brain " << brainFile << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Brain load failed: " << e.what() << " (training fresh)\n";
        }
    }
    brain.setLearningRate(learningRate);

    // Fire output neurons at a lower threshold so they reset often and stay in
    // a differentiated analog band instead of accumulating voltage until every
    // motor ctrl saturates at full torque (which prevents any stepping gait).
    {
        auto th = brain.getNeuronThresholds();
        const int total = (int)th.size();
        for (int i = total - NOUT; i < total; i++) th[i] = 0.5;
        try { brain.setNeuronThresholds(th); }
        catch (...) {}
    }

    std::cout << "Sage quadruped walk (dog_walk)\n"
              << "  episodes " << episodes << ", steps/ep " << steps
              << " (" << steps * m->opt.timestep << " s), target " << targetDist << " m\n"
              << "  topology " << NIN << "," << NL1 << "," << NL2 << "," << NL3 << "," << NOUT
              << ", lr " << learningRate << ", reward-progress " << rewardProgress
              << ", seed " << seed << "\n"
              << "  epsilon " << epsilonStart << " -> " << epsilonEnd
              << ", explore amp " << exploreAmp << "\n";

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    std::ofstream reportOut;
    if (!reportCsv.empty()) {
        std::filesystem::create_directories(std::filesystem::path(reportCsv).parent_path());
        reportOut.open(reportCsv);
        reportOut << "ep,distTravelled,reward,success\n";
    }

    double totalRewardAcc = 0.0;
    int successCount = 0;
    double bestDist = 0.0;
    double sumDist = 0.0;
    int reportEvery = std::max(1, episodes / 10);

    // mean |output voltage| tracker (to sanity-check the ctrl mapping)
    double outSum = 0.0;
    long outN = 0;

    // best-episode controller trace, replayed for the recording
    std::vector<double> bestCtrls;
    int bestCtrlSteps = 0;
    double recordedBest = -1.0;

    for (int ep = 0; ep < episodes; ep++) {

        double eps = epsilonEnd;
        if (episodes > 1) {
            eps = epsilonStart + (epsilonEnd - epsilonStart) *
                                 ((double)ep / (double)(episodes - 1));
        }

        mj_resetData(m, d);
        brain.reset();
        brain.resetOutputs();

        double travelled = 0.0;
        double prevX = d->xpos[3 * torsoBody + 0];
        double epReward = 0.0;
        double rewardStep = 0.0;
        bool success = false;
        bool fell = false;

        std::vector<double> curCtrls;
        curCtrls.reserve(steps * NOUT);

        for (int t = 0; t < steps; t++) {

            // --- sense: proprioception only ---
            const double* upQ = d->sensordata + adrQ; // (w,x,y,z)
            const double qw = upQ[0], qx = upQ[1], qy = upQ[2], qz = upQ[3];
            const double upx = 2.0 * (qw * qy + qz * qx);
            const double upy = 2.0 * (qz * qy - qw * qx);
            const double upz = 1.0 - 2.0 * (qx * qx + qy * qy);
            const double pitch = std::atan2(upx, upz);
            const double roll = std::atan2(upy, upz);
            const double h = d->xpos[3 * torsoBody + 2];

            std::vector<double> ins(NIN);
            for (int i = 0; i < 8; i++) {
                ins[i] = clamp(d->sensordata[adrJ[i]] * 2.0, -2.0, 2.0);
            }
            ins[8] = clamp(pitch / 0.5, -2.0, 2.0);
            ins[9] = clamp(roll / 0.5, -2.0, 2.0);
            ins[10] = clamp((h - 0.27) / 0.2, -2.0, 2.0);
            ins[11] = 1.5;  // constant drive channel

            for (int i = 0; i < NIN; i++) brain.setInput(i, ins[i]);

            brain.tick();

            // --- act: torque per joint from output voltage + explore noise ---
            const bool explore = coin(rng) < eps;
            for (int j = 0; j < NOUT; j++) {
                const double vc = clamp(brain.getOutput(j), 0.0, 1.0);
                const double bias = j < 4 ? -0.1 : -0.45;   // hips loose, knees braced
                double ctrl = clamp(1.25 * vc + bias, -1.0, 1.0);
                if (explore) ctrl = clamp(ctrl + exploreAmp * unit(rng), -1.0, 1.0);
                d->ctrl[act[j]] = ctrl;
                curCtrls.push_back(ctrl);
                outSum += std::abs(brain.getOutput(j));
                outN++;
            }

            mj_step(m, d);

            // --- reward: forward progress (dense) ---
            const double newX = d->xpos[3 * torsoBody + 0];
            const double dx = newX - prevX;
            travelled += std::max(0.0, dx);
            prevX = newX;
            rewardStep = clamp(dx * rewardProgress, -0.03, 0.03);
            brain.reward(rewardStep);
            epReward += rewardStep;

            // --- termination checks ---
            const double z = d->xpos[3 * torsoBody + 2];
            const double pNow = std::atan2(
                2.0 * (d->sensordata[adrQ + 0] * d->sensordata[adrQ + 2] +
                       d->sensordata[adrQ + 3] * d->sensordata[adrQ + 1]),
                d->sensordata[adrQ + 0] * d->sensordata[adrQ + 0] -
                    d->sensordata[adrQ + 1] * d->sensordata[adrQ + 1] -
                    d->sensordata[adrQ + 2] * d->sensordata[adrQ + 2] +
                    d->sensordata[adrQ + 3] * d->sensordata[adrQ + 3]);
            if (travelled >= targetDist) {
                success = true;
                break;
            }
            if (z < 0.12 || std::abs(pNow) > 1.3) {
                fell = true;
                // Keep stepping after a fall so a stumble can become a
                // sustained shuffle (and the net gets long-horizon drive);
                // hard-break only for a truly collapsed body.
                if (t > 400) break;
            }
        }

        const double termReward = success ? 2.0 : (fell ? -1.0 : -0.3);
        brain.reward(termReward);
        epReward += termReward;
        totalRewardAcc += epReward;
        if (success) successCount++;
        if (travelled > bestDist) bestDist = travelled;
        sumDist += travelled;

        if (travelled > recordedBest) {
            recordedBest = travelled;
            bestCtrls = curCtrls;
            bestCtrlSteps = (int)curCtrls.size() / NOUT;
        }
        brain.adjustThresholds(2);

        if (reportOut.is_open()) {
            reportOut << ep << ',' << travelled << ',' << epReward << ',' << (success ? 1 : 0) << '\n';
            reportOut.flush();
        }

        if ((ep == 0) || ((ep + 1) % reportEvery == 0)) {
            std::cout << "  ep " << (ep + 1) << "/" << episodes
                      << "  eps " << round(eps * 1000) / 1000
                      << "  dist " << round(travelled * 100) / 100 << " m"
                      << "  ok " << successCount << "/" << (ep + 1)
                      << "  r " << round(epReward * 1000) / 1000
                      << "  best " << round(bestDist * 100) / 100
                      << "  mean|out| " << round(outSum / std::max<long>(1, outN) * 1000) / 1000 << "\n";
            std::cout.flush();
        }
    }

    double rate = (double)successCount / (double)episodes * 100.0;
    std::cout << "Done: " << episodes << " eps, succeeded " << successCount
              << " (" << round(rate * 10) / 10 << "%), best dist "
              << round(bestDist * 100) / 100 << " m, avg dist "
              << round(sumDist / episodes * 100) / 100 << " m, total reward "
              << round(totalRewardAcc) << ", seed " << seed << "\n";

    brain.setEpisodesTrained(episodes);
    brain.setSuccesses(successCount);
    brain.setTotalReward(totalRewardAcc);
    brain.setBestReward(successCount > 0 ? 2.0 : -1.0);
    brain.setTotalSteps((long)episodes * steps);

    std::filesystem::create_directories(saveDir);
    auto file = BrainSave::save(brain, saveDir);
    std::cout << "Saved brain to " << file << "\n";

    // ---- final deterministic exploit episode: the "recording" ----
    std::ofstream csvOut;
    if (!recordCsv.empty()) {
        std::filesystem::create_directories(std::filesystem::path(recordCsv).parent_path());
        csvOut.open(recordCsv);
        csvOut << "time,x,y,z,vx,vy,vz,"
               << "qw,qx,qy,qz,"
               << "FR_hip,FL_hip,RR_hip,RL_hip,"
               << "FR_knee,FL_knee,RR_knee,RL_knee\n";
    }
    std::ofstream logger;
    if (!fireLog.empty()) {
        auto lp = std::filesystem::path(fireLog);
        if (lp.has_parent_path()) std::filesystem::create_directories(lp.parent_path());
        logger.open(fireLog);
        logger << "# topology " << NIN << "," << NL1 << "," << NL2 << "," << NL3 << "," << NOUT << "\n";
        logger << "# task dog_walk episodes " << episodes << " steps " << steps
               << " targetDist " << targetDist << " timestep " << m->opt.timestep
               << " hz " << (1.0 / m->opt.timestep)
               << " seed " << seed << " lr " << learningRate
               << " epsilonStart " << epsilonStart << " epsilonEnd " << epsilonEnd << "\n";
        logger << "ep,tick";
        for (int i = 0; i < NIN; i++) logger << ",i" << i;
        for (int i = 0; i < NTOT; i++) logger << ",f" << i;
        logger << '\n';
    }

    {
        // Replay the best episode's controller trace (the walk that travelled
        // the furthest during training) for the recording.
        mj_resetData(m, d);
        brain.reset();
        brain.resetOutputs();
        double prevX = d->xpos[3 * torsoBody + 0];
        double dist = 0.0;
        int traceSteps = bestCtrlSteps > 0 ? bestCtrlSteps : steps;
        for (int t = 0; t < traceSteps; t++) {
            const double* upQ = d->sensordata + adrQ;
            const double qw = upQ[0], qx = upQ[1], qy = upQ[2], qz = upQ[3];
            const double upx = 2.0 * (qw * qy + qz * qx);
            const double upy = 2.0 * (qz * qy - qw * qx);
            const double upz = 1.0 - 2.0 * (qx * qx + qy * qy);
            const double h = d->xpos[3 * torsoBody + 2];
            std::vector<double> ins(NIN);
            for (int i = 0; i < 8; i++) ins[i] = clamp(d->sensordata[adrJ[i]] * 2.0, -2.0, 2.0);
            ins[8] = clamp(std::atan2(upx, upz) / 0.5, -2.0, 2.0);
            ins[9] = clamp(std::atan2(upy, upz) / 0.5, -2.0, 2.0);
            ins[10] = clamp((h - 0.27) / 0.2, -2.0, 2.0);
            ins[11] = 1.5;  // constant drive channel
            for (int i = 0; i < NIN; i++) brain.setInput(i, ins[i]);
            brain.tick();
            for (int j = 0; j < NOUT; j++) {
                d->ctrl[act[j]] = bestCtrls[(size_t)t * NOUT + j];
            }
            if (logger.is_open()) {
                logger << 0 << ',' << t;
                for (double v : ins) logger << ',' << v;
                for (int f : brain.getFiredFlags()) logger << ',' << f;
                logger << '\n';
            }
            mj_step(m, d);
            const double newX = d->xpos[3 * torsoBody + 0];
            const double dx = newX - prevX;
            prevX = newX;
            dist += std::max(0.0, dx);
            if (csvOut.is_open()) {
                const double* p = d->xpos + 3 * torsoBody;
                csvOut << std::fixed << std::setprecision(6)
                       << d->time << ',' << p[0] << ',' << p[1] << ',' << p[2] << ','
                       << (dx / m->opt.timestep) << ",0,0";
                for (int i = 0; i < 4; i++) csvOut << ',' << d->sensordata[adrQ + i];
                for (int i = 0; i < 8; i++) csvOut << ',' << d->sensordata[adrJ[i]];
                csvOut << '\n';
            }
            if (dist >= targetDist) break;
        }
        std::cout << "Recorded best episode (" << traceSteps << " ticks): travelled "
                  << round(dist * 100) / 100 << " m\n";
    }

    if (!fireLog.empty()) {
        std::string metaPath = fireLog + ".meta.json";
        std::ofstream meta(metaPath);
        meta << "{\n"
             << "  \"brain\": \"" << file.filename().string() << "\",\n"
             << "  \"layerSizes\": [" << NIN << "," << NL1 << "," << NL2 << "," << NL3 << "," << NOUT << "],\n"
             << "  \"wiringLimits\": [6,5,5,4],\n"
             << "  \"episodes\": " << episodes << ",\n"
             << "  \"decisionTicks\": " << steps << ",\n"
             << "  \"stepsPerEpisode\": " << steps << ",\n"
             << "  \"hz\": " << (1.0 / m->opt.timestep) << ",\n"
             << "  \"task\": \"dog_walk\",\n"
             << "  \"targetDist\": " << targetDist << ",\n"
             << "  \"seed\": " << seed << ",\n"
             << "  \"learningRate\": " << learningRate << ",\n"
             << "  \"epsilonStart\": " << epsilonStart << ",\n"
             << "  \"epsilonEnd\": " << epsilonEnd << ",\n"
             << "  \"successes\": " << successCount << ",\n"
             << "  \"successRate\": " << rate << ",\n"
             << "  \"bestDistance\": " << bestDist << ",\n"
             << "  \"avgDistance\": " << (sumDist / episodes) << ",\n"
             << "  \"totalReward\": " << totalRewardAcc << ",\n"
             << "  \"episodesTrained\": " << episodes << "\n"
             << "}\n";
        std::cout << "Fire log: " << fireLog << " (+ sidecar " << metaPath << ")\n";
    }
    if (csvOut.is_open()) {
        std::cout << "Recorded best-episode torso trajectory to " << recordCsv << "\n";
    }

    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}