#ifndef DMO_TEAM_H
#define DMO_TEAM_H

#include "environment.h"
#include "policy.h"
#include "team.h"
#include "evolutionary_utils.h"
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <unordered_set>
#include <functional>

class DMOTeam;

class DMOTeam {
    int numberOfGenerations;
    int numberOfEpisodes;
    int populationSize;
    unsigned int teamIDCounter; // Tracks the latest team ID
    std::vector<Environment> generateTestEnvironments
        (const std::string& filename);
public:
    std::string domainName;
    std::vector<Individual> population;
    DMOTeam(const std::string& filename);
    void evolve(const std::string& filename, const std::string& data_filename);
};
#endif // DMO_TEAM
