#include "gpc.h"
#include "data.h"

#define RESTLABEL -1;
class MGPC {
public:
	MGPC(int n_features, Label label=0, int active_set_size=20);

	void update(const Sample& s);
	Label predict(const SparseVector& features);
private:
	std::map<Label,GPC*> mgpc_map;
	Label label;
};

