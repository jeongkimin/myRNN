#pragma once
#include "unit.h"

class VRU : public unit {
public:
	DenseLayer* I=nullptr;
	DenseLayer* H=nullptr;
	Activation* A=nullptr;
	double* hidden=nullptr;

	VRU(){}
	VRU(int input_sz, int output_sz, string actv_fn) {
		create(input_sz, output_sz, actv_fn);
	}
	~VRU() {
		if (I != nullptr) {
			delete I;
			delete H;
			delete A;
		}
		km_1d::free(hidden);
	}
	void create(int input_sz, int output_sz, string actv_fn){
		set_io(input_sz, output_sz);
		I = new DenseLayer(input_size, output_size, true);
		H = new DenseLayer(output_size, output_size, false);
		hidden = km_1d::alloc(output_size);
		A = define(actv_fn, output_size);
		_in_ = I->input_container;
	}

	virtual void allocMemory(int _port_size_) {
		n_storage = _port_size_;
		I->allocMemory(n_storage);
		H->allocMemory(n_storage);
		A->allocMemory(n_storage);
	}

	virtual void storeAt(const int& port_id) {
		I->storeAt(port_id);
		H->storeAt(port_id);
		A->storeAt(port_id);
	}

	virtual void init_fptt() {
		km_1d::fill_zero(hidden, output_size);
	}
	
	virtual void init_bptt() {
		km_1d::fill_zero(hidden, output_size);
	}
	
	virtual void pointAt(const int& port_id) {
		I->pointAt(port_id);
		H->pointAt(port_id);
		A->pointAt(port_id);
	}

	virtual void accum_forward(double*& next_port) {}
	
	virtual void accum_backward(double* const& out_grad) {}
	
	virtual void overwrite_forward(double*& next_port) {

		I->overwrite_forward(A->_in_);
		H->charge(hidden); //H->_in_ <= hidden
		H->accum_forward(A->_in_);
		A->overwrite_forward(hidden);
		km_1d::copy(next_port, hidden, output_size);
	}
	
	virtual void overwrite_backward(double* const& out_grad) {
		km_1d::add(hidden, out_grad, output_size);
		A->overwrite_backward(hidden);
		H->overwrite_backward(A->_in_);
		I->overwrite_backward(A->_in_);
		km_1d::copy(hidden, H->_in_, output_size);
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		I->delegate(p_bag, g_bag, len_bag);
		H->delegate(p_bag, g_bag, len_bag);
	}
};

class Gate :public unit {
public:
	double* second_container=nullptr;
	double** second_storage=nullptr;
	double* _SECOND_STORAGE_PTR_=nullptr;
	Gate() {}
	Gate(int input_sz) {
		create(input_sz);
	}
	~Gate() {
		km_1d::free(second_container);
		km_2d::free(second_storage, input_size);

	}
	void create(int input_sz) {
		set_io(input_sz, input_sz);
		input_container = km_1d::alloc(input_size);
		second_container = km_1d::alloc(input_size);
	}

	virtual void allocMemory(int _port_size_) {
		n_storage = _port_size_;
		input_storage = km_2d::alloc(n_storage, input_size);
		second_storage = km_2d::alloc(n_storage, input_size);
	}

	virtual void storeAt(const int& port_id) {
		km_1d::copy(second_storage[port_id], second_container, input_size);
		km_1d::copy(input_storage[port_id], input_container, input_size);
	}

	virtual void pointAt(const int& port_id) {
		_STORAGE_PTR_ = input_storage[port_id];
		_SECOND_STORAGE_PTR_ = second_storage[port_id];
	}

	virtual void accum_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] += input_container[i] * second_container[i];
		}
	}

	virtual void accum_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			input_container[i] += out_grad[i] * _SECOND_STORAGE_PTR_[i];
			second_container[i] += out_grad[i] * _STORAGE_PTR_[i];
		}
	}

	virtual void overwrite_forward(double*& next_port) {
		for (int i = 0; i < input_size; i++) {
			next_port[i] = input_container[i] * second_container[i];
		}
	}

	virtual void overwrite_backward(double* const& out_grad) {
		for (int i = 0; i < input_size; i++) {
			input_container[i] = out_grad[i] * _SECOND_STORAGE_PTR_[i];
			second_container[i] = out_grad[i] * _STORAGE_PTR_[i];
		}
	}
};

class LSTM : public unit {
public:
	DenseLayer* FI = nullptr;	DenseLayer* FH = nullptr;
	Sigmoid* FA = nullptr;	Gate* forgot_gate = nullptr;
	DenseLayer* II = nullptr;	DenseLayer* IH = nullptr;
	Sigmoid* IA = nullptr;
	DenseLayer* CI = nullptr;	DenseLayer* CH = nullptr;
	Tanh* CA = nullptr;	Gate* input_gate = nullptr;

	DenseLayer* OI = nullptr;	DenseLayer* OH = nullptr;
	Sigmoid* OA = nullptr;
	Tanh* L2S_A = nullptr;
	Gate* long_short_gate = nullptr;


	double* long_term = nullptr;
	double* short_term = nullptr;

	LSTM() {}
	LSTM(int input_sz, int output_sz) {
		create(input_sz, output_sz);
	}
	~LSTM() {
		if (FI != nullptr) {
			delete FI; delete II; delete CI; delete OI;
			delete FH; delete IH; delete CH; delete OH;
			delete FA; delete IA; delete CA; delete OA;
			delete L2S_A; delete input_gate; delete forgot_gate;
			delete long_short_gate;
		}
		km_1d::free(long_term);
		km_1d::free(short_term);
	}
	void create(int input_sz, int output_sz) {
		set_io(input_sz, output_sz);
		long_term = km_1d::alloc(output_size);
		short_term = km_1d::alloc(output_size);
		FI = new DenseLayer(input_size, output_size, true);
		II = new DenseLayer(input_size, output_size, true);
		CI = new DenseLayer(input_size, output_size, true);
		OI = new DenseLayer(input_size, output_size, true);

		FH = new DenseLayer(output_size, output_size, true);
		IH = new DenseLayer(output_size, output_size, true);
		CH = new DenseLayer(output_size, output_size, true);
		OH = new DenseLayer(output_size, output_size, true);
		FA = new Sigmoid(output_size);
		IA = new Sigmoid(output_size);
		CA = new Tanh(output_size);
		OA = new Sigmoid(output_size);
		L2S_A = new Tanh(output_size);
		forgot_gate = new Gate(output_size);
		input_gate = new Gate(output_size);
		long_short_gate = new Gate(output_size);



		input_container = km_1d::alloc(input_size);
		_in_ = input_container;
	}

	virtual void allocMemory(int _port_size_) {
		n_storage = _port_size_;
		FI->allocMemory(n_storage); FH->allocMemory(n_storage);
		FA->allocMemory(n_storage); forgot_gate->allocMemory(n_storage);
		
		II->allocMemory(n_storage); IH->allocMemory(n_storage);
		IA->allocMemory(n_storage); CI->allocMemory(n_storage);
		CH->allocMemory(n_storage); CA->allocMemory(n_storage);
		input_gate->allocMemory(n_storage);

		OI->allocMemory(n_storage); OH->allocMemory(n_storage);
		OA->allocMemory(n_storage); L2S_A->allocMemory(n_storage);
		long_short_gate->allocMemory(n_storage);
		input_storage = km_2d::alloc(n_storage, input_size);

	} 

	virtual void storeAt(const int& port_id) {
		FI->storeAt(port_id); FH->storeAt(port_id);
		FA->storeAt(port_id); forgot_gate->storeAt(port_id);

		II->storeAt(port_id); IH->storeAt(port_id); 
		IA->storeAt(port_id); CI->storeAt(port_id);
		CH->storeAt(port_id); CA->storeAt(port_id);
		input_gate->storeAt(port_id);

		OI->storeAt(port_id); OH->storeAt(port_id);
		OA->storeAt(port_id); long_short_gate->storeAt(port_id);
		L2S_A->storeAt(port_id);
		km_1d::copy(input_storage[port_id], input_container, input_size);
	}

	virtual void init_fptt() {
		km_1d::fill_zero(short_term, output_size);
		km_1d::fill_zero(long_term, output_size);
	}

	virtual void init_bptt() {
		km_1d::fill_zero(short_term, output_size);
		km_1d::fill_zero(long_term, output_size);
	}

	virtual void pointAt(const int& port_id) {
		FI->pointAt(port_id); FH->pointAt(port_id);
		FA->pointAt(port_id); forgot_gate->pointAt(port_id);
		II->pointAt(port_id); IH->pointAt(port_id);
		IA->pointAt(port_id); CI->pointAt(port_id);
		CH->pointAt(port_id); CA->pointAt(port_id);
		input_gate->pointAt(port_id);

		OI->pointAt(port_id); OH->pointAt(port_id);
		OA->pointAt(port_id); long_short_gate->pointAt(port_id);
		L2S_A->pointAt(port_id);
		_STORAGE_PTR_ = input_storage[port_id];
	}

	virtual void accum_forward(double*& next_port) {}

	virtual void accum_backward(double* const& out_grad) {}

	virtual void overwrite_forward(double*& next_port) {
		FI->charge(_in_);	FH->charge(short_term);
		II->charge(_in_);	IH->charge(short_term);
		CI->charge(_in_);	CH->charge(short_term);
		OI->charge(_in_);	OH->charge(short_term);

		FI->overwrite_forward(FA->_in_); FH->accum_forward(FA->_in_);
		km_1d::copy(forgot_gate->input_container, long_term, output_size);	//forgotGate.first <= longterm
		FA->overwrite_forward(forgot_gate->second_container);	//forgotGate.second <= FA.out
		II->overwrite_forward(IA->_in_); IH->accum_forward(IA->_in_);
		CI->overwrite_forward(CA->_in_); CH->accum_forward(CA->_in_);
		IA->overwrite_forward(input_gate->input_container); //input.first <= IA.out
		CA->overwrite_forward(input_gate->second_container); //input.second <= CA.out
		forgot_gate->overwrite_forward(long_term);
		input_gate->accum_forward(long_term);
		L2S_A->charge(long_term);
		L2S_A->overwrite_forward(long_short_gate->input_container); //lsgate.first <= L2S.out

		OI->overwrite_forward(OA->_in_); OH->accum_forward(OA->_in_);
		OA->overwrite_forward(long_short_gate->second_container);//lsgate.second <= OA.out
		long_short_gate->overwrite_forward(short_term);
		km_1d::copy(next_port, short_term, output_size);
	}

	virtual void overwrite_backward(double* const& out_grad) {
		for (int i = 0; i < output_size; i++) {
			short_term[i] += out_grad[i];
		}
		long_short_gate->overwrite_backward(short_term);
		L2S_A->overwrite_backward(long_short_gate->input_container);
		for (int i = 0; i < output_size; i++) {
			long_term[i] += L2S_A->_in_[i];
		}
		OA->overwrite_backward(long_short_gate->second_container);
		OH->overwrite_backward(OA->_in_);
		OI->overwrite_backward(OA->_in_);

		input_gate->overwrite_backward(long_term);
		IA->overwrite_backward(input_gate->input_container);
		CA->overwrite_backward(input_gate->second_container);
		II->overwrite_backward(IA->_in_);	IH->overwrite_backward(IA->_in_);
		CI->overwrite_backward(CA->_in_);	CH->overwrite_backward(CA->_in_);
		forgot_gate->overwrite_backward(long_term);
		FA->overwrite_backward(forgot_gate->second_container);
		km_1d::copy(long_term, forgot_gate->input_container, output_size);
		FI->overwrite_backward(FA->_in_);
		FH->overwrite_backward(FA->_in_);
		for (int i = 0; i < input_size; i++) {
			_in_[i] = FI->_in_[i] + II->_in_[i] + CI->_in_[i] + OI->_in_[i];
		}
		for (int i = 0; i < output_size; i++) {
			short_term[i] = FH->_in_[i] + IH->_in_[i] + CH->_in_[i] + OH->_in_[i];
		}
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		FI->delegate(p_bag, g_bag, len_bag);
		FH->delegate(p_bag, g_bag, len_bag);
		II->delegate(p_bag, g_bag, len_bag);
		IH->delegate(p_bag, g_bag, len_bag);
		CI->delegate(p_bag, g_bag, len_bag);
		CH->delegate(p_bag, g_bag, len_bag);
		OI->delegate(p_bag, g_bag, len_bag);
		OH->delegate(p_bag, g_bag, len_bag);
	}
};

class Many2Many {
public:
	
	StackBundle* nn;
	int data_len=0;

	Many2Many() {

	}

	void FPTT(double** const& input, double**& output, int sq_len) {
		data_len = sq_len;
		nn->init_fptt();
		for (int t = 0; t < data_len; t++) {
			nn->charge(input[t]);
			nn->overwrite_forward(output[t]);
			nn->storeAt(t);
		}
	}

	void BPTT(double** const& out_grad) {
		nn->init_bptt();
		for (int t = data_len -1; t >= 0; t--) {
			nn->pointAt(t);
			nn->overwrite_backward(out_grad[t]);
		}
	}
};