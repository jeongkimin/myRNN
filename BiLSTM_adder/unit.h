#pragma once
#include "core.h"

class unit {
public:

	int input_size = 0; int output_size = 0;
	double* _in_ = nullptr;
	int n_storage=0;
	double* input_container = nullptr;
	double** input_storage = nullptr;
	double* _STORAGE_PTR_ = nullptr;
	unit() {}
	void set_io(const int& input_sz, const int& output_sz) { input_size = input_sz; output_size = output_sz; }
	virtual void allocMemory(int _port_size_) = 0;
	virtual void charge(double* const& _input) { km_1d::copy(_in_, _input, input_size); }
	virtual void storeAt(const int& port_id) {};
	virtual void pointAt(const int& port_id) {};
	virtual void init_fptt() {};
	virtual void init_bptt() {};
	virtual void accum_forward(double*& next_port) = 0;
	virtual void accum_backward(double* const& out_grad) = 0;
	virtual void overwrite_forward(double*& next_port) = 0;
	virtual void overwrite_backward(double* const& out_grad) = 0;
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {}
	~unit() {}
};

class DenseLayer : public unit {
public:

	double* weight = nullptr; double* w_grad = nullptr;
	double* bias = nullptr; double* b_grad = nullptr;
	int w_len = 0, b_len = 0;

	DenseLayer() {}
	DenseLayer(int input_sz, int output_sz, bool use_bias = true) { create(input_sz, output_sz, use_bias); }

	virtual void allocMemory(int _port_size_) {
		km_2d::free(input_storage, n_storage);
		n_storage = _port_size_;
		input_storage = km_2d::alloc(n_storage, input_size);
	}

	void create(int input_sz, int output_sz, bool USE_BIAS) {
		set_io(input_sz, output_sz);
		input_container = km_1d::alloc(input_size);
		_in_ = input_container;
		w_len = input_size * output_size;
		b_len = output_size;
		weight = km_1d::alloc(w_len); w_grad = km_1d::alloc(w_len);
		km_1d::fill_noise(weight, 0.0, 2.0 / (double)input_size, w_len);
		if (USE_BIAS) {
			bias = km_1d::alloc(b_len); b_grad = km_1d::alloc(b_len);
			km_1d::fill_zero(bias, b_len);
		}
	}
	virtual void accum_forward(double*& next_port) {
		if (bias != nullptr) {
			for (int w = 0; w < output_size; w++) {
				next_port[w] += bias[w];
			}
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				next_port[w] += weight[h * output_size + w] * _in_[h];
			}
		}
	}

	virtual void overwrite_forward(double*& next_port) {
		if (bias != nullptr) {
			for (int w = 0; w < output_size; w++) {
				next_port[w] = bias[w];
			}
		}
		else {
			for (int w = 0; w < output_size; w++) {
				next_port[w] = 0;
			}
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				next_port[w] += weight[h * output_size + w] * _in_[h];
			}
		}
	}

	virtual void accum_backward(double* const& out_grad) {
		if (bias != nullptr) {
			for (int w = 0; w < output_size; w++) {
				b_grad[w] += out_grad[w];
			}
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				w_grad[h * output_size + w] += out_grad[w] * _STORAGE_PTR_[h];
			}
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				_in_[h] += out_grad[w] * weight[h * output_size + w];
			}
		}
	}

	virtual void overwrite_backward(double* const& out_grad) {
		if (bias != nullptr) {
			for (int w = 0; w < output_size; w++) {
				b_grad[w] += out_grad[w];
			}
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				w_grad[h * output_size + w] += out_grad[w] * _STORAGE_PTR_[h];
			}
		}
		for (int h = 0; h < input_size; h++) {
			_in_[h] = 0;
			for (int w = 0; w < output_size; w++) {
				_in_[h] += out_grad[w] * weight[h * output_size + w];
			}
		}
	}

	void storeAt(const int& port_id) {
		km_1d::copy(input_storage[port_id], _in_, input_size);
	}

	virtual void pointAt(const int& port_id) {
		_STORAGE_PTR_ = input_storage[port_id];
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		p_bag.push_back(weight);
		g_bag.push_back(w_grad);
		len_bag.push_back(w_len);
		if (bias != nullptr) {
			p_bag.push_back(bias);
			g_bag.push_back(b_grad);
			len_bag.push_back(b_len);
		}
	}

	~DenseLayer() {
		km_1d::free(bias); km_1d::free(b_grad);
		km_1d::free(weight); km_1d::free(w_grad);
		km_2d::free(input_storage, n_storage);
	}
};

class Activation : public unit {
public:

	Activation() {}
	Activation(int size) {}
	virtual void allocMemory(int _port_size_) {
		km_2d::free(input_storage, n_storage);
		n_storage = _port_size_;
		input_storage = km_2d::alloc(n_storage, input_size);
	}
	~Activation() {
		km_2d::free(input_storage, n_storage);
		km_1d::free(input_container);
	}
};

class Sigmoid : public Activation {
public:
	Sigmoid() { }
	Sigmoid(int size) { create(size); }

	void create(int size) {
		set_io(size, size);
		input_container = km_1d::alloc(input_size);
		_in_ = input_container;
	}

	virtual void storeAt(const int& port_id) {
		km_1d::copy(input_storage[port_id], _in_, input_size);
	}

	virtual void pointAt(const int& port_id) {
		_STORAGE_PTR_ = input_storage[port_id];
	}

	virtual void accum_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] += 1.0 / (1.0 + exp(-_in_[i]));
		}
	}

	virtual void overwrite_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] = 1.0 / (1.0 + exp(-_in_[i]));
		}
	}

	virtual void accum_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			_in_[i] += out_grad[i] * ( (1.0 / (1.0 + exp(-_STORAGE_PTR_[i]))) *(1.0 - (1.0 / (1.0 + exp(-_STORAGE_PTR_[i])))));
		}
	}

	virtual void overwrite_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			_in_[i] = out_grad[i] * ((1.0 / (1.0 + exp(-_STORAGE_PTR_[i]))) * (1.0 - (1.0 / (1.0 + exp(-_STORAGE_PTR_[i])))));
		}
	}
};

class Tanh : public Activation {
public:
	Tanh() { }
	Tanh(int size) { create(size); }

	void create(int size) {
		set_io(size, size);
		input_container = km_1d::alloc(input_size);
		_in_ = input_container;
	}

	virtual void storeAt(const int& port_id) {
		km_1d::copy(input_storage[port_id], _in_, input_size);
	}

	virtual void pointAt(const int& port_id) {
		_STORAGE_PTR_ = input_storage[port_id];
	}

	virtual void accum_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] += (2.0 / (1.0 + exp(-2.0 * _in_[i]))) - 1.0;
		}
	}

	virtual void overwrite_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] = (2.0 / (1.0 + exp(-2.0 * _in_[i]))) - 1.0;
		}
	}

	virtual void accum_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			_in_[i] += (1.0 - _square_((2.0 / (1.0 + exp(-2.0 * _STORAGE_PTR_[i]))) - 1.0)) * out_grad[i];
		}
	}

	virtual void overwrite_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			_in_[i] = (1.0 - _square_((2.0 / (1.0 + exp(-2.0 * _STORAGE_PTR_[i]))) - 1.0)) * out_grad[i];
		}
	}
};

class ReLU : public Activation {
public:
	ReLU() { }
	ReLU(int size) { create(size); }

	void create(int size) {
		set_io(size, size);
		input_container = km_1d::alloc(input_size);
		_in_ = input_container;
	}

	virtual void storeAt(const int& port_id) {
		km_1d::copy(input_storage[port_id], _in_, input_size);
	}

	virtual void pointAt(const int& port_id) {
		_STORAGE_PTR_ = input_storage[port_id];
	}

	virtual void accum_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			if (_in_[i] >= 0) {
				next_port[i] += _in_[i];
			}
		}
	}

	virtual void overwrite_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			if (_in_[i] >= 0) {
				next_port[i] = _in_[i];
			}
			else {
				next_port[i] = 0.0;
			}
		}
	}

	virtual void accum_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			_in_[i] += out_grad[i];
		}
	}

	virtual void overwrite_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			if (_STORAGE_PTR_[i] >= 0) {
				_in_[i] = out_grad[i];
			}
			else {
				_in_[i] = 0.0;
			}
		}
	}
};


Activation* define(string fn_id, int IOsize) {
	if (fn_id == "Sigmoid") {
		return new Sigmoid(IOsize);
	}
	else if (fn_id == "Tanh") {
		return new Tanh(IOsize);
	}
	else {
		return new ReLU(IOsize);
	}
}

class n_Layer {
public:
	int n;
	n_Layer() { n = 0; }
	n_Layer(int _n) { n = _n; }
};

class StackBundle : public unit {
public:
	unit** layer = nullptr;
	int n_layer = 0;
	bool published = false;
	StackBundle() { }
	~StackBundle() {
		if (published) {
			for (int i = 0; i < n_layer; i++) {
				delete layer[i];
			}
			delete[] layer;
		}
		else if (layer == nullptr) {
			delete[] layer;
		}
	}

	StackBundle(n_Layer number_of_layers, int input_sz, int output_sz) {
		n_layer = number_of_layers.n;
		layer = new unit * [n_layer];
		input_size = input_sz;
		output_size = output_sz;
	}

	virtual void publish() { //set link
		for (int i = 0; i < n_layer - 1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		_in_ = layer[0]->_in_;
		published = true;
	}

	virtual void allocMemory(int num_of_port) {
		n_storage = num_of_port;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->allocMemory(n_storage);
		}
	}

	virtual void accum_forward(double*& next_port) {
		for (int i = 0; i < n_layer-1; i++) {
			layer[i]->overwrite_forward(layer[i+1]->_in_);
		}
		layer[n_layer - 1]->accum_forward(next_port);
	}

	virtual void overwrite_forward(double*& next_port) {
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->overwrite_forward(layer[i+1]->_in_);
		}
		layer[n_layer - 1]->overwrite_forward(next_port);
	}

	virtual void accum_backward(double* const& out_grad) {
		layer[n_layer - 1]->overwrite_backward(out_grad);
		for (int i = n_layer - 2; i >= 1; i--) {
			layer[i]->overwrite_backward(layer[i + 1]->_in_);
		}
		layer[0]->accum_backward(layer[1]->_in_);
	}

	virtual void overwrite_backward(double* const& out_grad) {
		layer[n_layer - 1]->overwrite_backward(out_grad);
		for (int i = n_layer - 2; i >= 0; i--) {
			layer[i]->overwrite_backward(layer[i + 1]->_in_);
		}
	}

	void storeAt(const int& port_id) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->storeAt(port_id);
		}
	}

	virtual void init_fptt() {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->init_fptt();
		}
	}

	virtual void init_bptt() {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->init_bptt();
		}
	}

	virtual void pointAt(const int& port_id) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->pointAt(port_id);
		}
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->delegate(p_bag, g_bag, len_bag);
		}
	}
};