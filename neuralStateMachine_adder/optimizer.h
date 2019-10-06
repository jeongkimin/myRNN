#pragma once
#include "unit.h"


class optimizer {
public:
	bool optim_alloced;
	double** params;
	double** grad;
	int* n_params;
	int n_bundle;
	double learning_rate;

	bool use_rms = false;
	double** RMS;
	double rms_accum;
	double RMS_BETA;

	bool use_momentum = false;
	double** MOMENTUM;
	double momentum_accum;
	double MOMENTUM_BETA;

	optimizer(unit* trainee) {
		optim_alloced = false;
		collect(trainee);
	}

	void use_RMSProp(const double _RMSBeta) {
		use_rms = true;
		RMS_BETA = _RMSBeta;
		RMS = new double* [n_bundle];
		for (int i = 0; i < n_bundle; i++) {
			RMS[i] = new double[n_params[i]];
		}
		reset();
		rms_accum = 1.0;
	}

	void use_Momentum(const double _MOMENTUMBeta) {
		use_momentum = true;
		MOMENTUM_BETA = _MOMENTUMBeta;
		MOMENTUM = new double* [n_bundle];
		for (int i = 0; i < n_bundle; i++) {
			MOMENTUM[i] = new double[n_params[i]];
		}
		reset();
		momentum_accum = 1.0;
	}

	void use_AdaptiveMomentum(const double _Momentum, const double _Rms) {
		use_Momentum(_Momentum);
		use_RMSProp(_Rms);
	}

	void reset() {
		if (use_rms) {
			for (int i = 0; i < n_bundle; i++) {
				km_1d::fill_zero(RMS[i], n_params[i]);
			}
			rms_accum = 1.0;
		}
		if (use_momentum) {
			for (int i = 0; i < n_bundle; i++) {
				km_1d::fill_zero(MOMENTUM[i], n_params[i]);
			}
			momentum_accum = 1.0;
		}
	}
	void zero_grad() {
		for (int i = 0; i < n_bundle; i++) {
			for (int j = 0; j < n_params[i]; j++) {
				grad[i][j] = 0.0;
			}
		}
	}
	void setLearingRate(const double& lr) {
		learning_rate = lr;
	}
	void step() {
		if (use_momentum && use_rms) {
			momentum_accum *= MOMENTUM_BETA;
			rms_accum *= RMS_BETA;
			for (int i = 0; i < n_bundle; i++) {
				for (int j = 0; j < n_params[i]; j++) {
					MOMENTUM[i][j] = MOMENTUM_BETA * MOMENTUM[i][j] + (1.0 - MOMENTUM_BETA) * grad[i][j];
					RMS[i][j] = RMS_BETA * RMS[i][j] + (1.0 - RMS_BETA) * grad[i][j] * grad[i][j];
					params[i][j] -= (learning_rate / sqrt((RMS[i][j] / (1.0 - rms_accum)) + epsilon)) * (MOMENTUM[i][j] / (1.0 - momentum_accum));
				}
			}

		}
		else if (use_momentum) {
			momentum_accum *= MOMENTUM_BETA;
			for (int i = 0; i < n_bundle; i++) {
				for (int j = 0; j < n_params[i]; j++) {
					MOMENTUM[i][j] = MOMENTUM_BETA * MOMENTUM[i][j] + (1.0 - MOMENTUM_BETA) * grad[i][j];
					params[i][j] -= learning_rate * MOMENTUM[i][j] / (1.0 - momentum_accum);
				}
			}
		}
		else if (use_rms) {
			rms_accum *= RMS_BETA;
			double denominater = 1.0 - rms_accum;
			for (int i = 0; i < n_bundle; i++) {
				for (int j = 0; j < n_params[i]; j++) {
					RMS[i][j] = RMS_BETA * RMS[i][j] + (1.0 - RMS_BETA) * grad[i][j] * grad[i][j];
					params[i][j] -= (learning_rate / sqrt((RMS[i][j] / (1.0 - rms_accum)) + epsilon)) * grad[i][j];
				}
			}
		}
		else {
			for (int i = 0; i < n_bundle; i++) {
				for (int j = 0; j < n_params[i]; j++) {
					params[i][j] -= learning_rate * grad[i][j];
				}
			}
		}
	}

	void collect(unit* trainee) {
		if (optim_alloced) {
			delete[] params;
			delete[] grad;
			delete[] n_params;
		}
		vector<double*> p_bag;
		vector<double*> g_bag;
		vector<int> len_bag;
		trainee->delegate(p_bag, g_bag, len_bag);
		n_bundle = p_bag.size();
		params = new double* [n_bundle];
		grad = new double* [n_bundle];
		n_params = new int[n_bundle];
		for (int n = 0; n < n_bundle; n++) {
			n_params[n] = len_bag[n];
			params[n] = p_bag[n];
			grad[n] = g_bag[n];
		}
		optim_alloced = true;
	}

	~optimizer() {
		if (use_momentum) {
			for (int i = 0; i < n_bundle; i++) {
				km_1d::free(MOMENTUM[i]);
			}
			delete[] MOMENTUM;
		}
		if (use_rms) {
			for (int i = 0; i < n_bundle; i++) {
				km_1d::free(RMS[i]);
			}
			delete[] RMS;
		}
		if (optim_alloced) {
			delete[] params;
			delete[] grad;
			delete[] n_params;
		}
	}
};