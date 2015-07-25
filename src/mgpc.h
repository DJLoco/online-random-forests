#include "gpc.h"
#include "data.h"
#include "classifier.h"
#include "utilities.h"
//  #define RESTLABEL -1;
class MGPC: public Classifier {
public:
	MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures, const Label &label=0, int active_set_size=20);
    MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures);
		
	virtual void update(Sample &s);
	Label predict(const SparseVector &features);
	
	virtual void train(DataSet &dataset);
	
	virtual Result eval(Sample &sample);
	virtual vector<Result> test(DataSet &dataset);
	virtual vector<Result> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts);
	
private:
	const int *m_numClasses;
	const int *restLabel;
	const Hyperparameters *m_hp;

	std::map<Label, GPC*> mgpc_map;
	
	Label m_label;
	Label label;
};

