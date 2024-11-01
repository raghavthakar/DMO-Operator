#ifndef MO_ROVER_H
#define MO_ROVER_H

#include <utility>
#include <vector>
#include <string>

// Point of Interest (POI) class definition
class POI {
    bool observed; // flag that tracks whether this POI has been observed
public:
    int id;
    int classId;
    double x, y; // Coordinates of the POI
    double observationRadius; // Observation radius of the POI
    int coupling; // Minimum number of agents needed in vicinity to generate reward 
    int reward; // Generated by each POI if coupling criteria is met
    bool eternalPOI; // Whether the POI is observable throughout the episode, or only for a window
    std::pair<int, int> observableWindow; // The window in which the POI is observable
    bool exactCouplingNeeded; // Does coupling need to be exactly met, or can a greater number of agents also recieve a reward?
    bool rewardOnce; // If POI only rewards for the first timestep it is observed, and deactivates afterwards

    POI(int id, int classId, double x, double y, 
        double observationRadius, int coupling, 
        int reward, std::pair<int, int> observableWindow, 
        bool exactCouplingNeeded, bool rewardOnce);
    
    bool isObserved();
    void setAsObserved();
};

// MORover class definition
class MORover {
private:
    std::vector<POI> pois; // Vector to store POIs
    int xLength, yLength; // Environment dimensions
    int penalty; // Penalty at each timestep

public:
    // Method to load configuration from YAML file
    void loadConfig(const std::string& filename);

    // Method to return the POIs in the environment
    std::vector<POI> getPOIs();

    // Method to return the dimensions of the environment
    std::pair<int, int> getDimensions();

    // move the agents and return the new position
    std::pair <double, double> moveAgent(std::pair<double, double> currentPos, std::pair<double, double> delta, double maxStepSize);

    // compute the rewards generated by the provided agents configuration
    std::vector<int> getRewards(std::vector<std::pair<double, double>> agentPositions, int stepNumber);

    // observations of an agent
    std::vector<double> getAgentObservations(std::pair<double, double> agentPos, int numberOfSensors, double observationRadius, std::vector<std::pair<double, double>> agentPositions);

    // Method to print information about loaded POIs
    void printInfo() const;

    // Clear all the stored POIs and set other members to 0
    void reset();
};

#endif // MO_ROVER_H
