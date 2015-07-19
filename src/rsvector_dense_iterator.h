#include <gmm/gmm_vector.h>

double default_value = 0.0;

struct rsvector_dense_iterator {
	typedef gmm::rsvector_iterator<double> SIT;
	typedef double              value_type;
	typedef value_type*         pointer;
	typedef value_type&         reference;
	typedef size_t              size_type;
	typedef ptrdiff_t           difference_type;
	typedef std::bidirectional_iterator_tag iterator_category;
	typedef rsvector_dense_iterator iterator;

	SIT it;
	size_type pos;

	reference operator *() const { return (pos != it.index()) ? default_value : *it; }
	pointer operator->() const { return &(operator*()); }

	iterator &operator ++() { if(++pos > it.index()) ++it; return *this; }
	iterator operator ++(int) { iterator tmp = *this; ++(*this); return tmp; }
	iterator &operator --() { if(--pos > (--it).index()) ++it; return *this; }
	iterator operator --(int) { iterator tmp = *this; --(*this); return tmp; }

	bool operator ==(const iterator &i) const { return it == i.it && pos == i.pos; }
	bool operator !=(const iterator &i) const { return !(i == *this); }

	size_type index(void) const { return pos; }
	rsvector_dense_iterator(void) { pos = 0; }
	rsvector_dense_iterator(const SIT &i) : it(i) { pos = 0; }
};

struct rsvector_const_dense_iterator {
	typedef gmm::rsvector_const_iterator<double> SIT;
	typedef double              value_type;
	typedef const value_type*   pointer;
	typedef const value_type&   reference;
	typedef size_t              size_type;
	typedef ptrdiff_t           difference_type;
	typedef std::forward_iterator_tag iterator_category;
	typedef rsvector_const_dense_iterator iterator;

	SIT it;
	size_type pos;

	reference operator *() const { return (pos != it.index()) ? default_value : *it; }
	pointer operator->() const { return &(operator*()); }
	size_type index(void) const { return pos; }

	iterator &operator ++() { if(++pos > it.index()) ++it; return *this; }
	iterator operator ++(int) { iterator tmp = *this; ++(*this); return tmp; }
	iterator &operator --() { if(--pos > (--it).index()) ++it; return *this; }
	iterator operator --(int) { iterator tmp = *this; --(*this); return tmp; }

	bool operator ==(const iterator &i) const { return it == i.it && pos == i.pos; }
	bool operator !=(const iterator &i) const { return !(i == *this); }

	rsvector_const_dense_iterator(void) { pos = 0; }
	rsvector_const_dense_iterator(const rsvector_dense_iterator &i) : it(i.it) { pos = 0; }
	rsvector_const_dense_iterator(const SIT &i) : it(i) { pos = 0; }
};

