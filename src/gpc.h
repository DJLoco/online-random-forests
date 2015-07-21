#ifndef GPC_HPP
#define GPC_HPP

#include "data.h"

#include <gp-lvm/CKern.h>
#include <gp-lvm/CNoise.h>
#include <gp-lvm/CIvm.h>

#include <vector>


typedef enum {
	INIT,
	TRAIN
} gpc_state;


class GPC {
public:
	GPC(int n_features, Label unclassified=0, int active_set_size=20);

	void update(const Sample& s);
	Label predict(const SparseVector& features);
	double likelihood(Label prediction, const SparseVector& features);
private:
	gpc_state state;

	// parameters for the gaussian process implementation

	// dimension of one sample
	int input_dim;

	int active_set_size;
	int select_crit;

	CKern* kernel;
	CNoise* noise;
	CIvm* predictor;

	std::vector<Sample>* buffered_samples;

	// hash map from the labels to their number of occurances
	// it is only used during the initialization phase for choosing
	// the labels that occur most often
	std::map<Label,int>* label_counter;

	Label label1, label2;
	Label unclassified;

	void get_training_matrices(CMatrix*& training_labels, CMatrix*& training_features);
	void choose_labels_from_buffer();
	bool is_pure();
};

#endif

