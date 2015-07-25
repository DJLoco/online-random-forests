#ifndef HYPERPARAMETERS_H_
#define HYPERPARAMETERS_H_

#include <string>
using namespace std;

class Hyperparameters
{
 public:
    Hyperparameters();
    Hyperparameters(const string& confFile);

    // Online node
    int numRandomTests;
    int numProjectionFeatures;
    int counterThreshold;
    int maxDepth;

    // Online tree

    // Online forest
    int numTrees;
    int useSoftVoting;
    int numEpochs;
	
	// Gaussian Process
	int activeSetSize;
	int maxIters;
	int kernIters;
	int noiseIters;

    // Data
    string trainData;
    string trainLabels;
    string testData;
    string testLabels;

    int numTrain;
    int numTest;

    // Output
    int verbose;
};

#endif /* HYPERPARAMETERS_H_ */
