#include "environment.h"
#include "MORover.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <yaml-cpp/yaml.h>

void Environment::loadConfig(const std::string& filename) {
    this->rover_env.loadConfig(filename);
}

// compute the rewards generated by the provided agents configuration
std::vector<int> Environment::getRewards(std::vector<std::pair<double, double>> agentPositions,
                                        int stepNumber) {
    return this->rover_env.getRewards(agentPositions, stepNumber);
}

// take in an agent's position and return new position based on environmental limits
std::pair<double, double> Environment::moveAgent(std::pair<double, double> currentPos, std::pair<double, double> delta, double maxStepSize) {
    int environmentXLength = this->rover_env.getDimensions().first;
    int environmentYLength = this->rover_env.getDimensions().second;

    double posX = currentPos.first;
    double posY = currentPos.second;

    double dx = delta.first  ;
    double dy = delta.second ;

    // dx, dy are between -1 and 1. max step here is sqrt(2), which should corresond to step of maxStepSize

    double scaleFactor = maxStepSize / sqrt(2);
    dx *= scaleFactor;
    dy *= scaleFactor;

    // Calculate the new position within environment limits
    double step_slope = dy / dx;

    if (posX + dx > environmentXLength){
        dx = environmentXLength - posX;
        dy = dx * step_slope;
        step_slope = dy / dx;;
    } else if (posX + dx < 0) {
        dx = -posX;
        dy = dx * step_slope;
        step_slope = dy / dx;;
    }

    if (posY + dy > environmentYLength){
        dy = environmentYLength - posY;
        dx = dy /step_slope;
        step_slope = dy / dx;;
    } else if (posY + dy < 0) {
        dy = -posY;
        dx = dy / step_slope;
        step_slope = dy / dx;;
    }

    // Update the agent's position
    posX += dx;
    posY += dy;

    // return the updated position
    return std::make_pair(posX, posY);
}

// observations of an agent
std::vector<double> Environment::getAgentObservations(std::pair<double, double> agentPos, int numberOfSensors, double observationRadius, std::vector<std::pair<double, double>> agentPositions) {
    return this->rover_env.getAgentObservations(agentPos, numberOfSensors, observationRadius, agentPositions);
}

std::pair<int, int> Environment::getDimensions() {
    return this->rover_env.getDimensions();
}

void Environment::reset() {
    this->rover_env.reset();
}
