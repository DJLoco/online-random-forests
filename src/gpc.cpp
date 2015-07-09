#include "gpc.hpp"

#include <gp-lvm/CMatrix.h>

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


GPC::GPC(int n_features, double label) {
	input_dim = n_features;
	target_label = label;

	active_set_size = 20;
	select_crit = CIvm::ENTROPY;

	// noise will be initialized in the training routine
	// s.t. the target can be set
	noise = (CNoise*) NULL;
	kernel = new CRbfKern( input_dim );

	CDist* prior = new CGammaDist();
	prior->setParam(1.0, 0);
	prior->setParam(1.0, 1);

	kernel->addPrior(prior,1);
}

void GPC::train(double *labels, double *features, int n_samples) {
	CMatrix* training_data = new CMatrix(n_samples, input_dim, features);
	CMatrix* training_labels = extract_label(new CMatrix(n_samples, 1, labels), target_label);

	// initialize the noise model if this is the first training
	// otherwise update the "target"
	if(noise == NULL) {
		noise = new CProbitNoise( training_labels );
	}
	else {
		CMatrix noiseParams(1, noise->getNumParams());
		noise->getParams( noiseParams );
		noise->setTarget( training_labels );
		noise->setParams( noiseParams );
	}

	predictor = new CIvm(training_data, training_labels, kernel, noise, select_crit, active_set_size, 3);
	predictor->optimise();
}

double GPC::predict(double *features) {
	CMatrix ft(1, input_dim, features);
	CMatrix pred(1,1);

	predictor->out(pred, ft);
	return pred.getVal(0,0);
}

void GPC::test(double* labels, double* features, int n_samples) {
	CMatrix* testing_labels = extract_label(new CMatrix(n_samples, 1, labels), target_label);
	CMatrix* testing_data = new CMatrix(n_samples, input_dim, features);

	predictor->test(*testing_labels, *testing_data);
}

