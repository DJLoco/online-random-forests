#ifndef GPC_HPP
#define GPC_HPP

#include "data.h"

#include <gp-lvm/CKern.h>
#include <gp-lvm/CNoise.h>
#include <gp-lvm/CIvm.h>

#include <vector>

class GPC {
public:
	GPC(int n_features, int active_set_size=20, unsigned int max_iters=0, unsigned int kern_iters=0, unsigned int noise_iters=0);

	void update(const Sample& s);
	Label predict(const SparseVector& features);
	double likelihood(const SparseVector& features);
private:
	// parameters for the gaussian process implementation

	// dimension of one sample
	int input_dim;

	int active_set_size;
	int select_crit;

	CKern* kernel;
	CNoise* noise;
	CIvm* predictor;

	bool default_optimization_params;
	unsigned int max_iters;
	unsigned int kern_iters;
	unsigned int noise_iters;

	std::vector<Sample>* buffered_samples;

	// hash map from the labels to their number of occurances
	// it is only used during the initialization phase for choosing
	// the labels that occur most often
	std::map<Label,int>* label_counter;

	void get_training_matrices(CMatrix*& training_labels, CMatrix*& training_features);
	void choose_labels_from_buffer();
	bool is_pure();
};

#endif

