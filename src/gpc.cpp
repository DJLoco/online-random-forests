#include "gpc.h"

#include <gp-lvm/CMatrix.h>

// for 'rsvector_iterator'
#include <gmm/gmm_vector.h>


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

void GPC::choose_labels(Label& label1, Label& labels2) {
	int* counter = new int[n_buffered_samples];

	for(int i=0; i<n_buffered_samples; i++) {
		counter[ (int) buffered_labels->getVal(i,0) ]++;
	}

	int max1 = unclassified;
	int max2 = unclassified;

	for(int i=0; i<n_buffered_samples; i++) {
		if(max1 == unclassified || counter[i] > counter[max1]) {
			max2 = max1;
			max1 = i;
		}
		else if(max2 == unclassified || counter[i] > counter[max2]) {
			max2 = i;
		}
	}

	label1 = max1;
	label2 = max2;
}


GPC::GPC(int n_features, Label unclassified, int active_set_size) {
	input_dim = n_features;

	this->unclassified = unclassified;
	this->active_set_size = active_set_size;

	select_crit = CIvm::ENTROPY;

	n_buffered_samples = 0;
	buffered_labels = new CMatrix(active_set_size + 1, 1);
	buffered_features = new CMatrix(active_set_size + 1, n_features);

	// noise will be initialized in the training routine
	// s.t. the target can be set
	noise = (CNoise*) NULL;
	kernel = new CRbfKern( input_dim );

	CDist* prior = new CGammaDist();
	prior->setParam(1.0, 0);
	prior->setParam(1.0, 1);

	kernel->addPrior(prior,1);
}


void GPC::update(Sample s) {
	// label
	buffered_labels->setVal((double) s.y, n_buffered_samples, 0);
	
	// features
	for(int i=0; i<input_dim; i++) {
		buffered_features->setVal(s.x[i], n_buffered_samples, i);
	}

	n_buffered_samples++;

	// do we have enough samples for training?
	if(n_buffered_samples > active_set_size + 1) {

		// do we already have a classifier?
		if( predictor == NULL ) {
			// no noise model existing yet => create new noise model
			noise = new CProbitNoise( buffered_labels );

			choose_labels(label1, label2);
		}
		else {
			// noise model is already existing => adapt model
			CMatrix noiseParams(1, noise->getNumParams());
			noise->getParams( noiseParams );
			noise->setTarget( buffered_labels );
			noise->setParams( noiseParams );
		}

		CMatrix* filtered_labels = extract_two_labels(buffered_labels, label1, label2);

		// update the GP
		predictor = new CIvm(buffered_features, filtered_labels, kernel, noise, select_crit, active_set_size, 3);
		predictor->optimise();

		delete filtered_labels;

		// reset the counter
		n_buffered_samples = 0;
	}


}

Label GPC::predict(const SparseVector& features) {
	vector<double> feature_vec(
		       gmm::rsvector_const_iterator<double>(features.begin()),
		       gmm::rsvector_const_iterator<double>(features.end())
	);

	if(predictor != NULL) {
		CMatrix ft(1, input_dim, feature_vec);
		CMatrix pred(1,1);

		predictor->out(pred, ft);

		return pred.getVal(0,0);
	}
	else {
		return unclassified;
	}
}

