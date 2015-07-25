#include "gpc.h"
#include "data.h"
#include "classifier.h"
#include "utilities.h"
//  #define RESTLABEL -1;
class MGPC: public Classifier {
public:
	MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures, const Label &label);
    MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures);
		
	virtual void update(Sample &s);
	
	virtual void train(DataSet &dataset);
	
	virtual Result eval(Sample &sample);
	virtual vector<Result> test(DataSet &dataset);
	virtual vector<Result> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts);
	
private:
	const int *m_numClasses;
	const Hyperparameters *m_hp;

	std::map<Label, GPC*> mgpc_map;
	
	const Label *m_label;
};

