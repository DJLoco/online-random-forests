#ifndef GPC_HPP
#define GPC_HPP

#include <gp-lvm/CKern.h>
#include <gp-lvm/CNoise.h>
#include <gp-lvm/CIvm.h>

class GPC {
public:
	GPC(int n_features, double label);

	void train(double* labels, double* features, int n_samples);
	double predict(double* features);
	void test(double* labels, double* features, int n_samples);
private:
	int input_dim;
	int active_set_size;
	int select_crit;
	double target_label;

	CKern* kernel;
	CNoise* noise;
	CIvm* predictor;
};

#endif
