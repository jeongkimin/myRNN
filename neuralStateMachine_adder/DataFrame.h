#pragma once
#include "core.h"
#include <cstdlib>

class DataSet {
public:
	int n_rows;
	int n_cols;
	double* max;
	double* min;
	double** data; bool alloced;
	DataSet(string path, int row_start, int col_start) {
		alloced = false;
		load(path, row_start, col_start);
	}

	void load(string path, int row_from, int column_from) {
		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			string trash;
			for (int i = 0; i < row_from; i++) {
				getline(fin, trash);
			}
			string line;
			vector<vector<double>> bag;
			while (fin.good()) {
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				string value_buffer;
				vector<double> line_data;
				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						line_data.push_back(atof(value_buffer.c_str()));
						value_buffer.clear();
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
				bag.push_back(line_data);
			}
			fin.close();
			n_rows = bag.size();
			n_cols = bag[0].size() - column_from;
			data = new double* [n_rows];
			for (int r = 0; r < n_rows; r++) {
				data[r] = new double[n_cols];
				for (int c = 0; c < n_cols; c++) {
					data[r][c] = bag[r][c + column_from];
				}
			}
			alloced = true;
		}
	}
	void print() {
		for (int i = 0; i < n_rows; i++) {
			for (int j = 0; j < n_cols; j++) {
				cout << data[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	void mnist_scaling(double min, double max) {
		double range = max - min;
		for (int i = 0; i < n_rows; i++) {
			for (int j = 0; j < n_cols; j++) {
				data[i][j] = data[i][j] * ((max - min) / 255.0) + min;
			}
		}
	}
	void scaling(double min, double max) {
		double origin_min = 0;
		double origin_max = 0;
		for (int c = 0; c < n_cols; c++) {
			origin_min = data[0][c];
			origin_max = data[0][c];
			for (int r = 1; r < n_rows; r++) {
				if (origin_max < data[r][c]) {
					origin_max = data[r][c];
				}
				if (origin_min > data[r][c]) {
					origin_min = data[r][c];
				}
			}
			for (int r = 0; r < n_rows; r++) {
				data[r][c] = (data[r][c] - origin_min) * ((max - min) / (origin_max - origin_min)) + min;
			}

		}
	}
	~DataSet() {
		if (alloced) {
			for (int i = 0; i < n_rows; i++) {
				delete[] data[i];
			}
			delete[] data;
		}
	}
};
class scaler {
public:
	double* origin_max;
	double* origin_min;
	int n_cols;
	double scaled_max;
	double scaled_min;
	scaler(int feature_size) {
		n_cols = feature_size;
		origin_max = new double[n_cols];
		origin_min = new double[n_cols];
	}
	void get_minmax(double** data, int n_rows) {

		for (int c = 0; c < n_cols; c++) {
			origin_min[c] = data[0][c];
			origin_max[c] = data[0][c];
			for (int r = 1; r < n_rows; r++) {
				if (origin_max[c] < data[r][c]) {
					origin_max[c] = data[r][c];
				}
				if (origin_min[c] > data[r][c]) {
					origin_min[c] = data[r][c];
				}
			}
			//cout << "min : " << origin_min[c] << ", max : " << origin_max[c] << endl;
		}

	}
	void scale(double** target, int n_rows, double _scaled_min, double _scaled_max) {
		scaled_max = _scaled_max;
		scaled_min = _scaled_min;
		for (int c = 0; c < n_cols; c++) {
			for (int r = 0; r < n_rows; r++) {
				target[r][c] = (target[r][c] - origin_min[c]) * ((scaled_max - scaled_min) / (origin_max[c] - origin_min[c])) + scaled_min;
			}
		}
	}
	void descale(double** target, int n_rows) {
		for (int c = 0; c < n_cols; c++) {
			for (int r = 0; r < n_rows; r++) {
				target[r][c] = (target[r][c] - scaled_min) * ((origin_max[c] - origin_min[c]) / (scaled_max - scaled_min)) + origin_min[c];
			}
		}
	}
	~scaler() {
		delete[] origin_max;
		delete[] origin_min;
	}
};
class Labelset {
public:
	int n_rows;
	int* label; bool alloced;
	Labelset() {}
	Labelset(string path, int row_start) {
		alloced = false;
		load(path, row_start);
	}
	void load(string path, int row_from) {
		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			string trash;
			for (int i = 0; i < row_from; i++) {
				getline(fin, trash);
			}
			string line;
			vector<int> bag;
			while (fin.good()) {
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				string value_buffer;
				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						bag.push_back(atoi(value_buffer.c_str()));
						value_buffer.clear();
						break;
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
			}
			fin.close();
			n_rows = bag.size();
			label = new int[n_rows];
			for (int i = 0; i < n_rows; i++) {
				label[i] = bag[i];
			}
			alloced = true;
			int min = label[0];
			for (int i = 1; i < n_rows; i++) {
				if (label[i] < min) {
					min = label[i];
				}
			}
			for (int i = 0; i < n_rows; i++) {
				label[i] -= min;
			}
		}
	}
	void print() {
		for (int i = 0; i < n_rows; i++) {
			cout << label[i] << endl;
		}
		cout << endl;
	}


	~Labelset() {
		if (alloced) {
			delete[] label;
		}
	}
};


class batch_loader {
public:
	DataSet* x_data;
	Labelset* y_data;
	double** mini_x;
	int* mini_y;
	int batch_size;
	int set_size;
	int x_feature_size;
	bool has_x;
	bool has_y;
	bool batch_memory_allocated;

	int nb_batch;
	int curr_order;

	int* order_list;
	batch_loader(DataSet* target_data, Labelset* label_data) {
		x_data = target_data;
		y_data = label_data;
		if (x_data->n_rows != y_data->n_rows) {
			cout << "?" << endl;
		}
		set_size = x_data->n_rows;
		x_feature_size = x_data->n_cols;
		has_x = true;
		has_y = true;
		batch_memory_allocated = false;

		order_list = new int[set_size];
		for (int i = 0; i < set_size; i++) {
			order_list[i] = i;
		}
		shuffle_order();
	}
	batch_loader(DataSet* target_data) {
		x_data = target_data;
		set_size = x_data->n_rows;
		x_feature_size = x_data->n_cols;
		has_x = true;
		has_y = false;
		batch_memory_allocated = false;
	}
	void shuffle_order() {
		km::shuffle(order_list, set_size);
	}
	void alloc_batch_storage(int new_batch_size) {
		if (batch_memory_allocated) {
			if (has_x) {
				km_2d::free(mini_x, batch_size);
			}
			if (has_y) {
				delete[] mini_y;
			}
		}
		batch_size = new_batch_size;
		curr_order = -1;
		nb_batch = set_size / batch_size;

		if (has_x) {
			mini_x = km_2d::alloc(batch_size, x_feature_size);
			batch_memory_allocated = true;
		}
		if (has_y) {
			mini_y = new int[batch_size];
			batch_memory_allocated = true;
		}
		next_batch();
	}
	void next_batch() {
		curr_order++;
		int step = curr_order * batch_size;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < x_feature_size; i++) {
				mini_x[m][i] = x_data->data[order_list[step + m]][i];
			}
			if (has_y) {
				mini_y[m] = y_data->label[order_list[m + step]];
			}
		}
		if (curr_order >= nb_batch - 1) {
			curr_order = -1;
			shuffle_order();
		}
	}
	void cur_batch_show() {
		int step = curr_order * batch_size;
		cout << "batch_idx : " << curr_order << endl;
		for (int i = 0; i < batch_size; i++) {
			cout << order_list[step + i] << " : ";
			for (int j = 0; j < x_feature_size; j++) {
				cout << mini_x[i][j] << " ";
			}
			if (has_y) {
				cout << "y : " << mini_y[i] << endl;
			}
		}
	}
	~batch_loader() {
		if (batch_memory_allocated) {
			if (has_x) {
				km_2d::free(mini_x, batch_size);
			}
			if (has_y) {
				delete[] mini_y;
			}
		}
		delete[] order_list;
	}
};