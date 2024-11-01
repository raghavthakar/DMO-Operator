#include "MORover.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

POI::POI(int id, int classId, double x, double y, double observationRadius, int coupling, 
    int reward, std::pair<int, int> observableWindow, bool exactCouplingNeeded, bool rewardOnce)
    : id(id), classId(classId), x(x), y(y), observationRadius(observationRadius), coupling(coupling), 
    reward(reward), observableWindow(observableWindow), exactCouplingNeeded(exactCouplingNeeded),
    rewardOnce(rewardOnce) {
        this->observed = false;
    }

bool POI::isObserved() {
    return this->observed;
}
void POI::setAsObserved() {
    this->observed = true;
}

void MORover::loadConfig(const std::string& filename) {
    YAML::Node config = YAML::LoadFile(filename); // Parse YAML from file

    const YAML::Node& dimensions = config["MORover"]["dimensions"];
    xLength = dimensions["xLength"].as<double>();
    yLength = dimensions["yLength"].as<double>(); // Read the MORover dimensions into the object

    penalty = config["MORover"]["penalty"].as<int>();

    int numberOfPOIs = config["MORover"]["numberOfPOIs"].as<int>();
    int numberOfClassIDs = config["MORover"]["numberOfClassIds"].as<int>();

    int poisPerClass = numberOfPOIs / numberOfClassIDs;
    int remainingPOIs = numberOfPOIs % numberOfClassIDs;

    bool randomPOIConfig = config["MORover"]["randomPOIConfig"].as<bool>();

    // Load up an MORover configuration

    // Generate and add random POIs if true
    if (randomPOIConfig == true) {
        int poi_num = 0; // Initialize poi_num outside the loop
        for (int classID = 0; classID < numberOfClassIDs; ++classID) {
            int poisToAdd = poisPerClass + (classID < remainingPOIs ? 1 : 0);
            for (int i = 0; i < poisToAdd; ++i) {
                // Random POI position
                double poi_x = rand() % xLength;
                double poi_y = rand() % yLength;

                std::pair<int, int> observableWindow;

                if (config["MORover"]["eternalPOIs"].as<bool>() == true) {
                    // set observable window to eternity
                    observableWindow = std::make_pair(0, std::numeric_limits<int>::max());
                }
                else {
                    observableWindow = std::make_pair(config["MORover"]["observableWindow"][0].as<int>(), 
                                                    config["MORover"]["observableWindow"][1].as<int>());
                }

                // Put into list
                pois.emplace_back(poi_num++, classID, poi_x, poi_y, 
                    config["MORover"]["observationRadius"].as<double>(),
                    config["MORover"]["coupling"].as<int>(),
                    config["MORover"]["reward"].as<int>(),
                    observableWindow,
                    config["MORover"]["exactCouplingNeeded"].as<bool>(),
                    config["MORover"]["rewardOnce"].as<bool>()); // Create POI object and add to vector
            }
        }
    }
    // Else, read the configs from poi list in config and add pois
    else {
        for (const auto& poi: config["MORover"]["POIs"]) {
            int poi_id = poi["id"].as<int>();
            if (poi_id > numberOfPOIs+1 || poi_id < 0) {
                std::cout<<"Invalid POI config"<<std::endl;
                exit(1);
            }

            int class_id = poi["classID"].as<int>();
            if (class_id > numberOfClassIDs+1 || class_id < 0) {
                std::cout<<"Invalid POI config"<<std::endl;
                exit(1);
            }

            double poi_x = poi["poi_x"].as<double>();
            double poi_y = poi["poi_y"].as<double>();
            if (poi_x > xLength || poi_x < 0 || poi_y > yLength || poi_y < 0) {
                std::cout<<"Invalid POI config"<<std::endl;
                exit(1);
            }

            std::pair<int, int> observableWindow;
            if (poi["eternalPOI"].as<bool>() == true) {
                // set observable window to eternity
                observableWindow = std::make_pair(0, std::numeric_limits<int>::max());
            }
            else {
                observableWindow = std::make_pair(poi["observableWindow"][0].as<int>(), 
                                                    poi["observableWindow"][1].as<int>());
            }



            pois.emplace_back(poi_id, class_id, poi_x, poi_y, 
                    poi["observationRadius"].as<double>(),
                    poi["coupling"].as<int>(),
                    poi["reward"].as<int>(),
                    observableWindow, poi["exactCouplingNeeded"].as<bool>(),
                    poi["rewardOnce"].as<bool>()); // Create POI object and add to vector
        }
    }
    
}

// move the agents and return the new position
std::pair <double, double> MORover::moveAgent(std::pair<double, double> currentPos, std::pair<double, double> delta, double maxStepSize) {
    int environmentXLength = this->getDimensions().first;
    int environmentYLength = this->getDimensions().second;

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

// compute the rewards generated by the provided agents configuration
std::vector<int> MORover::getRewards(std::vector<std::pair<double, double>> agentPositions,
                                        int stepNumber) {
    std::vector<int> rewardVector;

    // Determine the number of unique class IDs
    std::unordered_set<int> uniqueClassIds;
    for(const auto& poi : pois) {
        uniqueClassIds.insert(poi.classId);
    }
    int numberOfPOIClasses = uniqueClassIds.size();

    // As many elements in the reward vector as objectives (POI classes)
    for(int i = 0; i < numberOfPOIClasses; i++)
        rewardVector.push_back(0);
    
    // loop through each POI, and add its reward accordingly
    for (auto& poi : pois) {
        int numberOfCloseAgents = 0;

        for (const auto agentPosition : agentPositions) {
            double posX = agentPosition.first;
            double posY = agentPosition.second;

            double dx = poi.x - posX;
            double dy = poi.y - posY;
            double distance = sqrt(dx * dx + dy * dy);

            if(distance <= poi.observationRadius)
                numberOfCloseAgents++;
        }

        // get rewards from this POI if it is not observed, or if it is not rewardOnce POI
        if (!poi.isObserved() || (!poi.rewardOnce)) {
            // increase the rewards if agent within POI's observation radius & 
            // timestep within the POI's observableWindow
            if (poi.exactCouplingNeeded == true && numberOfCloseAgents == poi.coupling && 
                stepNumber >= poi.observableWindow.first && stepNumber <= poi.observableWindow.second) {
                rewardVector[poi.classId] += poi.reward;
                // set this POI as observed
                poi.setAsObserved();
            }
            else if (poi.exactCouplingNeeded == false && numberOfCloseAgents >= poi.coupling && 
                stepNumber >= poi.observableWindow.first && stepNumber <= poi.observableWindow.second) {
                rewardVector[poi.classId] += poi.reward;
                // set this POI as observed
                poi.setAsObserved();
            }
        }
    }

    // Add in the penalties of each agent to each objective reward
    for (int i = 0; i < rewardVector.size(); i++)
        rewardVector[i] += agentPositions.size() * penalty;

    return rewardVector;
}

// observations of an agent
std::vector<double> MORover::getAgentObservations(std::pair<double, double> agentPos, int numberOfSensors, double observationRadius, std::vector<std::pair<double, double>> agentPositions) {
    std::vector<double> observations; // To store the observations the agent makes

    double posX = agentPos.first;
    double posY = agentPos.second;

    observations.push_back(posX);
    observations.push_back(posY);

    // Get the POI observations
    // Determine the number of unique class IDs
    std::unordered_set<int> uniqueClassIds;
    for(const auto& poi : this->getPOIs()) {
        uniqueClassIds.insert(poi.classId);
    }
    int numberOfPOIClasses = uniqueClassIds.size();

    int* POIObservations = new int[numberOfPOIClasses * numberOfSensors]; // Store POI observations excusively
    // Initialize all elements to zero
    for (int i = 0; i < numberOfPOIClasses * numberOfSensors; ++i) {
        POIObservations[i] = 0;
    }

    for(auto& poi : getPOIs()) {
        // Calculate the angle between the central point and point of interest
        double dx = poi.x - posX;
        double dy = poi.y - posY;
        double angle = atan2(dy, dx) * 180 / M_PI; // Convert to degrees
        double distance = sqrt(dx * dx + dy * dy);

        // Normalize the angle to be in the range [0, 360)
        if (angle < 0) {
            angle += 360;
        }

        // Calculate the angle between each cone boundary
        double coneAngle = 360.0 / numberOfSensors;

        // Check which cone the point lies inside
        for (int i = 0; i < numberOfSensors; ++i) {
            double coneStart = i * coneAngle;
            double coneEnd = (i + 1) * coneAngle;
            
            // Adjust for negative angles
            coneStart = (coneStart < 0) ? coneStart + 360 : coneStart;
            coneEnd = (coneEnd < 0) ? coneEnd + 360 : coneEnd;
            
            // Check if the angle falls within the current cone
            if (coneStart <= angle && angle <= coneEnd && distance <= observationRadius) {
                POIObservations[poi.classId * numberOfSensors + i]++; // Incremenet obs at cone index
            }
        }
    }

    int* agentObservations = new int[numberOfSensors]; // Store the other agents observations
    // Initialize all elements to zero
    for (int i = 0; i < numberOfSensors; ++i) {
        agentObservations[i] = 0;
    }

    for (auto& otherAgentPosition : agentPositions) {
        // Calculate the angle between the central point and point of interest
        double dx = otherAgentPosition.first - posX;
        double dy = otherAgentPosition.second - posY;

        // Do not process the same agent in the observation
        if (dx == 0 && dy == 0)
            continue;
        
        double angle = atan2(dy, dx) * 180 / M_PI; // Convert to degrees
        double distance = sqrt(dx * dx + dy * dy);

        // Normalize the angle to be in the range [0, 360)
        if (angle < 0) {
            angle += 360;
        }

        // Calculate the angle between each cone boundary
        double coneAngle = 360.0 / numberOfSensors;

        // Check which cone the point lies inside
        for (int i = 0; i < numberOfSensors; ++i) {
            double coneStart = i * coneAngle;
            double coneEnd = (i + 1) * coneAngle;
            
            // Adjust for negative angles
            coneStart = (coneStart < 0) ? coneStart + 360 : coneStart;
            coneEnd = (coneEnd < 0) ? coneEnd + 360 : coneEnd;
            
            // Check if the angle falls within the current cone
            if (coneStart <= angle && angle <= coneEnd && distance <= observationRadius) {
                agentObservations[i]++; // Incremenet obs at cone index
            }
        }
    }

    // Append the POIObservations array to the observations vector
    for (int i = 0; i < numberOfPOIClasses * numberOfSensors; ++i) {
        observations.push_back(POIObservations[i]);
    }
    // Append the agentObservations array to the observations vector
    for (int i = 0; i < numberOfSensors; ++i) {
        observations.push_back(agentObservations[i]);
    }
    // Delete the dynamically allocated array to free memory
    delete[] POIObservations;
    delete[] agentObservations;
    
    return observations;
}

void MORover::printInfo() const {
    std::cout << "POIs in the MORover:" << std::endl;
    for (const auto& poi : pois) {
        std::cout << "ID: " << poi.id << ", Class: " << poi.classId 
        << ", Coordinates: (" << poi.x << ", " << poi.y << "), Observation Radius: " 
        << poi.observationRadius << ", ObsWindow: [" << poi.observableWindow.first << ","
        << poi.observableWindow.second << "]" << " Reward: " << poi.reward << std::endl;
    }
}

std::vector<POI> MORover::getPOIs() {
    return pois;
}

std::pair<int, int> MORover::getDimensions() {
    return std::make_pair(xLength, yLength);
}

void MORover::reset() {
    pois.clear();
    xLength = 0;
    yLength = 0;
    penalty = 0;
}
