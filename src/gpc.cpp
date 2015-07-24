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
		result->push_back(*iter);
	}

	return *result;
}

/**
 * Return a new label matrix that takes one label out of the others.
 *
 * The result will be a new matrix that has a 1 at every place where the given label occurred, otherwise -1.
 *
 * @param labels_matrix the matrix containing the original labels
 * @param filtered_label the label to pick
 * @return the new label matrix
 */
CMatrix *extract_label(CMatrix *labels_matrix, double filtered_label)
{
	CMatrix *result = new CMatrix(*labels_matrix);

	for(unsigned int i=0; i<result->getRows(); i++) {
		for(unsigned int j=0; j<result->getCols(); j++) {
			if(abs(result->getVal(i,j) - filtered_label) <= FLT_EPSILON)
				result->setVal(1.0,i,j);
			else
				result->setVal(-1.0,i,j);
		}
	}

	return result;
}

CMatrix* extract_two_labels(CMatrix* labels_matrix, double first_label, double second_label)
{
	CMatrix* result = new CMatrix(*labels_matrix);

	for(unsigned int i=0; i<result->getRows(); i++) {
		for(unsigned int j=0; j<result->getCols(); j++) {
			double val = result->getVal(i,j);
			if(abs(val - first_label) <= FLT_EPSILON)
				result->setVal(1.0,i,j);
			else if(abs(val - second_label) <= FLT_EPSILON)
				result->setVal(-1.0,i,j);
			else
				result->setVal(0.0,i,j);
		}
	}

	return result;
}

/**
 * Decide which two labels to classify using the data collected in the buffer.
 *
 * This routine determines which two labels occur most often in the collected
 * data and makes them being classified.
 */
void GPC::choose_labels_from_buffer() {
	// get the two most used labels
	bool max1_defined = false;
	bool max2_defined = false;

	Label max1 = unclassified;
	Label max2 = unclassified;

	for(std::map<Label,int>::iterator i=label_counter->begin(); i != label_counter->end(); i++) {
		if(!max1_defined || i->second > (*label_counter)[max1]) {
			max2 = max1;
			max1 = i->first;
			max1_defined = true;
		}
		else if(!max2_defined || i->second > (*label_counter)[max2]) {
			max2 = i->first;
			max2_defined = true;
		}
	}

	// set the chosen labels
	label1 = max1;
	label2 = max2;
}

bool GPC::is_pure() {
	return (label_counter->size() < 2);
}


GPC::GPC(int n_features, Label unclassified, int active_set_size, unsigned int max_iters, unsigned int kern_iters, unsigned int noise_iters) {
	state = TRAIN;
	input_dim = n_features;

	this->unclassified = unclassified;
	this->active_set_size = active_set_size;
	this->max_iters = max_iters;
	this->kern_iters = kern_iters;
	this->noise_iters = noise_iters;
	this->default_optimization_params = (noise_iters == 0);
	//
	this->label1 = unclassified;
	this->label2 = 5;

	select_crit = CIvm::ENTROPY;

	buffered_samples = new std::vector<Sample>();
	label_counter = new std::map<Label,int>();

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

	int n_samples = (*label_counter)[label1] + (*label_counter)[label2];

	training_labels = new CMatrix(n_samples, 1);
	training_features = new CMatrix(n_samples, input_dim);

	int row = 0;

	for(std::vector<Sample>::iterator it=buffered_samples->begin(); it != buffered_samples->end(); it++) {
		if(it->y != label1 && it->y != label2)
			continue;

		training_labels->setVal(it->y, row, 0);

		int col = 0;

		for(feature_it ft=feature_it(it->x.begin()); col < input_dim; ft++, col++) {
			training_features->setVal(*ft, row, col);
		}

		row++;
	}

	// TODO: memory leak: delete training_labels before overwriting
	training_labels = extract_two_labels(training_labels, label1, label2);
}


/**
 * Train the classifier with a new sample.
 *
 * We collect the training data, until the number of collected
 * samples exceeds the "active set size" of the gaussian process
 * classifier.
 * If we have already decided which labels to train, the input data
 * will be filtered immediately. Samples with other labels will be
 * thrown away.
 * If we have not yet decided which labels to train, we fill the buffer
 * matrix with all samples we get, then decide which labels to classify
 * and filter the data we got so far.
 *
 * @param s the sample to be trained
 */
void GPC::update(const Sample& s) {

	buffered_samples->push_back(s);

	// update the label counter
	if(label_counter->find(s.y) == label_counter->end()) {
		label_counter->insert(std::map<Label,int>::value_type(s.y,1));
	} else {
		(*label_counter)[s.y]++;
	}


	switch(state) {

	case INIT:
		if(!is_pure() && buffered_samples->size() > active_set_size) {
			choose_labels_from_buffer();
			state = TRAIN;
		}

		break;

	case TRAIN:
		// check if we have collected enough data to initiate training
		if(!is_pure() && (*label_counter)[label1] + (*label_counter)[label2] > active_set_size) {

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
			predictor = new CIvm(training_features, training_labels, kernel, noise, select_crit, active_set_size, 3);

			if(default_optimization_params)
				predictor->optimise();
			else
				predictor->optimise(max_iters, kern_iters, noise_iters);

			// reset the counters for the labels
			delete buffered_samples;
			delete label_counter;
			buffered_samples = new std::vector<Sample>();
			label_counter = new std::map<Label,int>();
		}

		break;
	}

}

Label GPC::predict(const SparseVector& features) {
	vector<double> feature_vec = to_dense_vector(features);

	if(predictor != NULL) {
		CMatrix ft(1, input_dim, feature_vec);
		CMatrix pred(1,1);

		predictor->out(pred, ft);

		return (pred.getVal(0,0) == 1) ? label1 : label2;
	}
	else {
		return unclassified;
	}
}

double GPC::likelihood(Label prediction, const SparseVector& features) {
	vector<double> feature_vec = to_dense_vector(features);

	CMatrix prob_mat(1,1);
	CMatrix result_mat(1,1);
	CMatrix feature_mat(1, input_dim, feature_vec);

	if(predictor != NULL) {
		predictor->out(result_mat, prob_mat, feature_mat);
		if(prediction == label1) {
			if(result_mat.getVal(0,0) - 1 < FLT_EPSILON && result_mat.getVal(0,0) - 1 > -FLT_EPSILON) {
				return prob_mat.getVal(0,0);
			}
			else {
				return 1 - prob_mat.getVal(0,0);
			}
		}
	}
	else {
		// no valid prediction possible => no likelihood!
		return 0;
	}

}

