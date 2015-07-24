#include "mgpc.h"

MGPC::MGPC(const int &numClasses, const int &numFeatures, const Label &label, int active_set_size):
	m_numClasses(&numClasses), restLabel(&numClasses) {
	this->label = label;
	this->m_label = label;

	cout << "--- Online Gaussian Process Initialization --- Label: " << m_label << " --- " << endl;
	for (Label i = 0; i < *m_numClasses; i++) {
		GPC *gpc = new GPC(numFeatures, i, active_set_size, 1, 1, 1);
		mgpc_map.insert(std::map<Label,GPC*>::value_type(i,gpc));
	}
}

MGPC::MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures):
	m_numClasses(&numClasses), restLabel(&numClasses), m_hp(&hp) {
	int active_set_size = 20;
	for (Label i = 0; i < *m_numClasses; i++) {
		GPC *gpc = new GPC(numFeatures, i, active_set_size, 1, 1, 1);
		mgpc_map.insert(std::map<Label,GPC*>::value_type(i,gpc));
	}
}

void MGPC::update(Sample &s) {
	for (Label i = 0; i < *m_numClasses; i++) {
		Sample sample;
		sample.x = s.x;
		
		if(s.y == i)
			sample.y = s.y;
		else
			sample.y = *restLabel;
	
		sample.w = 1.0;

		mgpc_map[i]->update(sample);
	}
}

Label MGPC::predict(const SparseVector& features) {
	int argmax = label;
	double max = 0;
	for (int i = 0; i < *m_numClasses; i++) {
		if(max < mgpc_map[i]->likelihood(i, features)) {
			max = mgpc_map[i]->likelihood(i, features);
			argmax = i;
		}
	}

	if(max < 0.00001) {
		cout << "--- Online Gaussian prediction error ---" << endl;
	}
	return argmax; // if there is no good prediction, it returns the default label
}

void MGPC::train(DataSet &dataset) {
	vector<int> randIndex;
	int sampRatio = dataset.m_numSamples / 10;
	for (int n = 0; n < m_hp->numEpochs; n++) {
		randPerm(dataset.m_numSamples, randIndex);
		for (int i = 0; i < dataset.m_numSamples; i++) {
			update(dataset.m_samples[randIndex[i]]);
			if (m_hp->verbose >= 3 && (i % sampRatio) == 0) {
				cout << "--- Online Gaussian Process training --- Epoch: " << n + 1 << " --- ";
				cout << (10 * i) / sampRatio << "%" << endl;
			}
		}
	}
}

Result MGPC::eval(Sample &sample) {
	Result result;
	vector<double> confidence;
	int prediction;
	int argmax = 0;
	double max = 0;
	for (int i = 0; i < *m_numClasses; i++) {
		double likelihood = mgpc_map[i]->likelihood(i, sample.x);
		confidence.push_back(likelihood);
		if(max < likelihood) {
			max = likelihood;
			argmax = i;
		}
	}

	if(max < 0.0001) {
		prediction = argmax;
	} else
		prediction = argmax;

	result.confidence = confidence;
	result.prediction = prediction;
	std::cout << "Prediction: " << prediction << ", Label: " << sample.y << ", confidence: " << confidence << "; 42" << std::endl;
	return result;
}

vector<Result> MGPC::test(DataSet &dataset) {
	vector<Result> results;
	for (int i = 0; i < dataset.m_numSamples; i++) {
		results.push_back(eval(dataset.m_samples[i]));
	}

	double error = compError(results, dataset);
	if (m_hp->verbose >= 3) {
		cout << "--- Online Random Tree test error: " << error << endl;
	}

	return results;
}

vector<Result> MGPC::trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts) {
	vector<Result> results;
	vector<int> randIndex;
	int sampRatio = dataset_tr.m_numSamples / 10;
	vector<double> testError;
	for (int n = 0; n < m_hp->numEpochs; n++) {
		randPerm(dataset_tr.m_numSamples, randIndex);
		for (int i = 0; i < dataset_tr.m_numSamples; i++) {
			update(dataset_tr.m_samples[randIndex[i]]);
			if (m_hp->verbose >= 3 && (i % sampRatio) == 0) {
				cout << "--- Online Gaussian Process training --- Epoch: " << n + 1 << " --- ";
				cout << (10 * i) / sampRatio << "%" << endl;
			}
		}

		results = test(dataset_ts);
		testError.push_back(compError(results, dataset_ts));
	}

	if (m_hp->verbose >= 3) {
		cout << endl << "--- Online Gaussian Process test error over epochs: ";
		dispErrors(testError);
	}

	return results;
}
