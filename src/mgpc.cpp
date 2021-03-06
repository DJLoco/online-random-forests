#include "mgpc.h"

MGPC::MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures, const Label &label, int active_set_size):
	m_numClasses(&numClasses) {
	this->label = label;
	this->m_label = label;

	cout << "--- Online Gaussian Process Initialization --- Label: " << m_label << " --- " << endl;
	for (Label i = 0; i < *m_numClasses; i++) {
		GPC *gpc = new GPC(numFeatures, hp.activeSetSize, hp.maxIters, hp.kernIters, hp.noiseIters);
		mgpc_map.insert(std::map<Label,GPC*>::value_type(i,gpc));
	}
}

MGPC::MGPC(const Hyperparameters &hp, const int &numClasses, const int &numFeatures):
	m_numClasses(&numClasses), m_hp(&hp) {
	this->m_label = 0;
	for (Label i = 0; i < *m_numClasses; i++) {
		GPC *gpc = new GPC(numFeatures, hp.activeSetSize, hp.maxIters, hp.kernIters, hp.noiseIters);
		mgpc_map.insert(std::map<Label,GPC*>::value_type(i,gpc));
	}
}

void MGPC::update(Sample &s) {
	for (Label i = 0; i < *m_numClasses; i++) {
		Sample sample;
		sample.x = s.x;
		
		if(s.y == i)
			sample.y = 1;
		else
			sample.y = -1;
	
		sample.w = 1.0;

		mgpc_map[i]->update(sample);
	}
}

Label MGPC::predict(const SparseVector& features) {
	int argmax = m_label;
	double max = 0;
	for (int i = 0; i < *m_numClasses; i++) {
		if(max < mgpc_map[i]->likelihood(features)) {
			max = mgpc_map[i]->likelihood(features);
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
	int argmax = m_label;
	double max = 0;

	for (int i = 0; i < *m_numClasses; i++) {
		double likelihood = mgpc_map[i]->likelihood(sample.x);
		confidence.push_back(likelihood);
		if(max < likelihood) {
			max = likelihood;
			argmax = i;
		}
	}

	result.confidence = confidence;
	result.prediction = argmax;
	std::cout << "Prediction: " << argmax << ", Label: " << sample.y << ", confidence: " << confidence << ";" << std::endl;
	return result;
}

vector<Result> MGPC::test(DataSet &dataset) {
	vector<Result> results;
	for (int i = 0; i < dataset.m_numSamples; i++) {
		results.push_back(eval(dataset.m_samples[i]));
	}

	double error = compError(results, dataset);
	if (m_hp->verbose >= 3) {
		cout << "--- Online Gaussian Process test error: " << error << endl;
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
