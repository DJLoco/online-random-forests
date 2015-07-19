#include "mgpc.h"

MGPC::MGPC(int n_features, Label label, int active_set_size) {
	this->label = label;
	for (Label i = 0; i < 5; i++) {
		GPC* gpc = new GPC(n_features, i, active_set_size);
		mgpc_map.insert(std::map<Label,GPC*>::value_type(i,gpc));
	}
}

void MGPC::update(const Sample& s) {	
	for (Label i = 0; i < 5; i++) {
		Sample sample;
		sample.x = s.x;
		
		if(s.y == i)
			sample.y = s.y;
		else
			sample.y = RESTLABEL;
	
		sample.w = 1.0;

		mgpc_map[i]->update(sample);
	}
}

Label MGPC::predict(const SparseVector& features) {
	// TODO
	return label;
}
