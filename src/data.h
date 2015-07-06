#ifndef DATA_H_
#define DATA_H_

#include <iostream>
#include <vector>
#include <gmm/gmm.h>
#include <string>

#include "hyperparameters.h"

using namespace std;
using namespace gmm;

// TYPEDEFS
typedef int Label;
typedef double Weight;
typedef rsvector<double> SparseVector;

// DATA CLASSES
class Sample {
  public:
    SparseVector x;
    Label y;
    Weight w;

    void disp() {
	cout << "Sample: y = " << y << ", w = " << w << ", x = ";
	cout << x << endl;
    }
};

class DataSet {
  private:
    void loadLIBSVM(string filename);
    void loadRGBD(string fileLabels, string fileData, int n_samples);
  public:
    vector<Sample> m_samples;
    int m_numSamples;
    int m_numFeatures;
    int m_numClasses;

    vector<double> m_minFeatRange;
    vector<double> m_maxFeatRange;

    void findFeatRange();

    void loadTrain(Hyperparameters hp);
    void loadTest(Hyperparameters hp);
};

class Result {
  public:
    vector<double> confidence;
    int prediction;
};

#endif /* DATA_H_ */
