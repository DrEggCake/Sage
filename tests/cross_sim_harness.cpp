#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>

#include <mujoco/mujoco.h>

struct Vec3 {
    double x, y, z;
};

struct TrajectoryPoint {
    double time;
    Vec3 position;
    Vec3 velocity;
    std::vector<double> jointPositions;
    std::vector<double> jointVelocities;
};

struct TestResult {
    std::string scenarioName;
    std::string simulatorName;
    std::vector<TrajectoryPoint> trajectory;
    Vec3 finalPosition;
    double totalEnergy;
    double finalKE;
    double finalPE;
    double maxDisplacement;
    bool passed;
    std::string notes;
};

int findBodyId(mjModel* m, const std::string& name) {
    for (int i = 0; i < m->nbody; i++) {
        if (std::string(m->names + m->name_bodyadr[i]) == name) {
            return i;
        }
    }
    return -1;
}

TestResult runMuJoCoTest(
    const std::string& modelPath,
    const std::string& scenarioName,
    const std::string& bodyName,
    int steps,
    double noiseStdDev = 0.0
) {
    char error[1000];
    mjModel* m = mj_loadXML(modelPath.c_str(), nullptr, error, 1000);
    if (!m) {
        std::cerr << "Failed to load model: " << error << "\n";
        return {scenarioName, "mujoco", {}, {0,0,0}, 0, 0, 0, 0, false, error};
    }

    mjData* d = mj_makeData(m);
    if (!d) {
        mj_deleteModel(m);
        return {scenarioName, "mujoco", {}, {0,0,0}, 0, 0, 0, 0, false, "Failed to create data"};
    }

    int bodyId = findBodyId(m, bodyName);
    if (bodyId < 0) {
        mj_deleteData(d);
        mj_deleteModel(m);
        return {scenarioName, "mujoco", {}, {0,0,0}, 0, 0, 0, 0, false,
                "Body '" + bodyName + "' not found in model"};
    }

    TestResult result;
    result.scenarioName = scenarioName;
    result.simulatorName = "mujoco";
    result.passed = true;

    double initialHeight = 0.0;
    double maxDisp = 0.0;

    for (int i = 0; i < steps; i++) {
        mj_step(m, d);

        TrajectoryPoint pt;
        pt.time = d->time;

        const double* pos = d->xpos + 3 * bodyId;
        const double* vel = d->cvel + 6 * bodyId;

        pt.position.x = pos[0];
        pt.position.y = pos[1];
        pt.position.z = pos[2];

        pt.velocity.x = vel[0];
        pt.velocity.y = vel[1];
        pt.velocity.z = vel[2];

        for (int j = 0; j < m->nq; j++) {
            pt.jointPositions.push_back(d->qpos[j]);
        }
        for (int j = 0; j < m->nv; j++) {
            pt.jointVelocities.push_back(d->qvel[j]);
        }

        result.trajectory.push_back(pt);

        if (i == 0) {
            initialHeight = pt.position.z;
        }

        double disp = std::sqrt(
            pt.position.x * pt.position.x +
            pt.position.y * pt.position.y +
            (pt.position.z - initialHeight) * (pt.position.z - initialHeight)
        );
        if (disp > maxDisp) maxDisp = disp;

        if (noiseStdDev > 0.0) {
            d->qpos[0] += noiseStdDev * ((double)rand() / RAND_MAX - 0.5);
            d->qpos[1] += noiseStdDev * ((double)rand() / RAND_MAX - 0.5);
        }
    }

    result.finalPosition = result.trajectory.back().position;
    result.maxDisplacement = maxDisp;

    result.finalKE = 0.0;
    for (int j = 0; j < m->nv; j++) {
        double mv = d->qvel[j];
        result.finalKE += 0.5 * mv * mv;
    }
    result.finalPE = result.finalPosition.z;
    result.totalEnergy = result.finalKE + result.finalPE;

    mj_deleteData(d);
    mj_deleteModel(m);

    return result;
}


void printResult(const TestResult& r) {
    std::cout << "=== " << r.scenarioName << " [" << r.simulatorName << "] ===\n";
    std::cout << "  Steps: " << r.trajectory.size() << "\n";
    std::cout << "  Final pos: (" << r.finalPosition.x << ", "
              << r.finalPosition.y << ", " << r.finalPosition.z << ")\n";
    std::cout << "  Max displacement: " << r.maxDisplacement << "\n";
    std::cout << "  Final KE: " << r.finalKE << ", PE: " << r.finalPE
              << ", Total: " << r.totalEnergy << "\n";
    std::cout << "  Passed: " << (r.passed ? "YES" : "NO") << "\n";
    if (!r.notes.empty()) {
        std::cout << "  Notes: " << r.notes << "\n";
    }
    std::cout << "\n";
}


void compareResults(const TestResult& mujoco, const TestResult& other) {
    std::cout << "=== COMPARISON: " << mujoco.scenarioName << " ===\n";

    double posError = std::sqrt(
        std::pow(mujoco.finalPosition.x - other.finalPosition.x, 2) +
        std::pow(mujoco.finalPosition.y - other.finalPosition.y, 2) +
        std::pow(mujoco.finalPosition.z - other.finalPosition.z, 2)
    );

    double energyError = std::abs(mujoco.totalEnergy - other.totalEnergy);
    double dispError = std::abs(mujoco.maxDisplacement - other.maxDisplacement);

    std::cout << "  Position error (L2): " << posError << "\n";
    std::cout << "  Energy error: " << energyError << "\n";
    std::cout << "  Max displacement error: " << dispError << "\n";

    bool match = posError < 0.05 && energyError < 0.1;
    std::cout << "  Match: " << (match ? "YES" : "MISMATCH") << "\n";
    std::cout << "\n";

    if (!match) {
        std::cout << "  *** FIRST MISMATCH RECORDED ***\n";
        std::cout << "  Position diff: (" << mujoco.finalPosition.x - other.finalPosition.x
                  << ", " << mujoco.finalPosition.y - other.finalPosition.y
                  << ", " << mujoco.finalPosition.z - other.finalPosition.z << ")\n";
        std::cout << "  This suggests: physics engine differences in "
                  << (posError > 0.1 ? "contact handling" : "numerical integration") << "\n\n";
    }
}


void saveTrajectory(const TestResult& r, const std::string& filename) {
    std::ofstream f(filename);
    f << "time,x,y,z,vx,vy,vz\n";
    for (const auto& pt : r.trajectory) {
        f << std::fixed << std::setprecision(6)
          << pt.time << ","
          << pt.position.x << "," << pt.position.y << "," << pt.position.z << ","
          << pt.velocity.x << "," << pt.velocity.y << "," << pt.velocity.z << "\n";
    }
}


std::string getModelPath(const std::string& name) {
    std::filesystem::path p = std::filesystem::current_path();
    while (p.parent_path() != p) {
        auto candidate = p / "tests" / "models" / (name + ".xml");
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
        p = p.parent_path();
    }
    auto fallback = std::filesystem::current_path() / "tests" / "models" / (name + ".xml");
    return fallback.string();
}


void runCartPoleTest() {
    std::cout << "\n### TEST 1: Cart-Pole Balancing ###\n\n";

    auto mujocoResult = runMuJoCoTest(
        getModelPath("cart_pole"),
        "cart_pole",
        "cart",
        500,
        0.0
    );
    printResult(mujocoResult);
    saveTrajectory(mujocoResult, "tests/results/cart_pole_mujoco.csv");

    std::cout << "MuJoCo cart-pole complete. Trajectory saved.\n";
    std::cout << "To compare with OmniSim, run the Python companion script.\n\n";
}


void runBallDropTest() {
    std::cout << "\n### TEST 2: Ball Drop with Friction ###\n\n";

    auto mujocoResult = runMuJoCoTest(
        getModelPath("ball_drop"),
        "ball_drop",
        "ball",
        2000,
        0.0
    );
    printResult(mujocoResult);
    saveTrajectory(mujocoResult, "tests/results/ball_drop_mujoco.csv");

    std::cout << "MuJoCo ball-drop complete. Trajectory saved.\n\n";
}


void runArmReachTest() {
    std::cout << "\n### TEST 3: Simple Arm Reach ###\n\n";

    auto mujocoResult = runMuJoCoTest(
        getModelPath("arm_reach"),
        "arm_reach",
        "ee",
        1000,
        0.0
    );
    printResult(mujocoResult);
    saveTrajectory(mujocoResult, "tests/results/arm_reach_mujoco.csv");

    std::cout << "MuJoCo arm-reach complete. Trajectory saved.\n\n";
}


int main(int argc, char** argv) {

    std::filesystem::create_directories("tests/results");

    std::cout << "========================================\n";
    std::cout << "  Sage Cross-Simulator Test Harness\n";
    std::cout << "  MuJoCo backend\n";
    std::cout << "========================================\n\n";

    std::string suite = (argc > 1) ? argv[1] : "all";

    if (suite == "all" || suite == "cart_pole") {
        runCartPoleTest();
    }
    if (suite == "all" || suite == "ball_drop") {
        runBallDropTest();
    }
    if (suite == "all" || suite == "arm_reach") {
        runArmReachTest();
    }

    std::cout << "========================================\n";
    std::cout << "  All MuJoCo tests complete.\n";
    std::cout << "  Results in tests/results/\n";
    std::cout << "  Run omnisim_compare.py for cross-sim comparison.\n";
    std::cout << "========================================\n";

    return 0;
}
