#ifndef GPC_HPP
#define GPC_HPP

#include "data.h"

#include <gp-lvm/CKern.h>
#include <gp-lvm/CNoise.h>
#include <gp-lvm/CIvm.h>


class GPC {
public:
	GPC(int n_features, Label unclassified=0, int active_set_size=20);

	void update(const Sample& s);
	Label predict(const SparseVector& features);
private:
	int input_dim;
	int active_set_size;
	int select_crit;

	int n_buffered_samples;
	CMatrix* buffered_labels, *buffered_features;

	Label label1, label2;
	Label unclassified;

	CKern* kernel;
	CNoise* noise;
	CIvm* predictor;

	void choose_labels(Label& label1, Label& label2);
};

#endif
