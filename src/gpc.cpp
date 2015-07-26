#include "gpc.h"

#include <gp-lvm/CMatrix.h>

// for 'rsvector_dense_iterator'
#include "rsvector_dense_iterator.h"

// for choosing the two most often occurring labels
#include <map>

typedef rsvector_dense_iterator feature_it;
typedef rsvector_const_dense_iterator const_feature_it;


vector<double>& to_dense_vector(const SparseVector& sv) {
	vector<double>* result = new vector<double>;

	int fi = 0;

	for(const_feature_it iter=const_feature_it(sv.begin()); fi<sv.size(); fi++, iter++) {
		result->push_back((double) *iter);
	}

	return *result;
}

bool GPC::is_pure() {
	if ((*label_counter)[1] > 0 && (*label_counter)[-1] > 0)
		return false;
	else
		return true;
}


GPC::GPC(int n_features, int active_set_size, unsigned int max_iters, unsigned int kern_iters, unsigned int noise_iters) {

	input_dim = n_features;

	this->active_set_size = active_set_size;
	this->max_iters = max_iters;
	this->kern_iters = kern_iters;
	this->noise_iters = noise_iters;
	this->default_optimization_params = (noise_iters == 0);

	select_crit = CIvm::ENTROPY;

	buffered_samples = new std::vector<Sample>();
	label_counter = new std::map<Label,int>();
	label_counter->insert(std::map<Label,int>::value_type(1,0));
	label_counter->insert(std::map<Label,int>::value_type(-1,0));

	// noise will be initialized in the training routine
	// s.t. the target can be set
	noise = (CNoise*) NULL;
	kernel = new CRbfKern( input_dim );

	CDist* prior = new CGammaDist();
	prior->setParam(1.0, 0);
	prior->setParam(1.0, 1);

	kernel->addPrior(prior,1);

	predictor = (CIvm*) NULL;
}


void GPC::get_training_matrices(CMatrix*& training_labels, CMatrix*& training_features) {

	int n_samples = (*label_counter)[1] + (*label_counter)[-1];

	training_labels = new CMatrix(n_samples, 1);
	training_features = new CMatrix(n_samples, input_dim);

	int row = 0;

	for(std::vector<Sample>::iterator it=buffered_samples->begin(); it != buffered_samples->end(); it++) {
		if(it->y != 1 && it->y != -1)
			continue;

		vector<double> featureVec = to_dense_vector(it->x);
		CMatrix *feature = new CMatrix(1, input_dim, featureVec);
		training_labels->setVal(it->y, row, 0);
		
		for(int i=0; i<input_dim; i++) {
			training_features->setVal(it->x.r(i), row, i);
		}

		row++;
	}
}


/**
 * Train the classifier with a new sample.
 *
 * We collect the training data, until the number of collected
 * samples exceeds the "active set size" of the gaussian process
 * classifier.
 *
 * @param s the sample to be trained
 */
void GPC::update(const Sample& s) {

	buffered_samples->push_back(s);

	if(s.y != 1 && s.y !=-1) {
		exit(EXIT_FAILURE);
	}
		

	// update the label counter
	(*label_counter)[s.y]++;

	// check if we have collected enough data to initiate training
	if(!is_pure() && buffered_samples->size() > active_set_size) {

		// copy the relevant samples into a matrix
		CMatrix* training_labels;
		CMatrix* training_features;
		get_training_matrices(training_labels, training_features);

		// is this the first learning cycle?
		if(predictor == NULL) {
			// no noise model existing yet => create new noise model
			noise = new CProbitNoise( training_labels );
		}
		else {
			// noise model is already existing => adapt model
			CMatrix noiseParams(1, noise->getNumParams());
			noise->getParams( noiseParams );
			noise->setTarget( training_labels );
			noise->setParams( noiseParams );
		}

		// update the gaussian process model
		predictor = new CIvm(training_features, training_labels, kernel, noise, select_crit, active_set_size, 0);

		predictor->setDefaultOptimiser(CIvm::SCG);
		// predictor->setDefaultOptimiser(CIvm::CG);   // conjgrad
		// predictor->setDefaultOptimiser(CIvm::GD);   // graddesc
		// predictor->setDefaultOptimiser(CIvm::BFGS); // quasinew

		predictor->optimise(max_iters, kern_iters, noise_iters);

		// reset the counters for the labels
		buffered_samples->clear();

		(*label_counter)[1] = 0;
		(*label_counter)[-1] = 0;
	}


}

Label GPC::predict(const SparseVector& features) {
	vector<double> feature_vec = to_dense_vector(features);

	if(predictor != NULL) {
		CMatrix ft(1, input_dim, feature_vec);
		CMatrix pred(1,1);

		predictor->out(pred, ft);

		return pred.getVal(0,0);
	}
	return 0;
}

double GPC::likelihood(const SparseVector& features) {
	vector<double> feature_vec = to_dense_vector(features);

	CMatrix prob_mat(1,1);
	CMatrix result_mat(1,1);
	CMatrix feature_mat(1, input_dim, feature_vec);

	if(predictor != NULL) {
		predictor->out(result_mat, prob_mat, feature_mat);

		return (((int) result_mat.getVal(0,0)) == 1) ? prob_mat.getVal(0,0) : 1 - prob_mat.getVal(0,0);
	}
	else {
		// no valid prediction possible => no likelihood!
		return 0;
	}

}

