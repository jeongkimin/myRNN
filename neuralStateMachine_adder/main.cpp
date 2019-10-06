#include "unit.h"
#include "optimizer.h"
#include "DataFrame.h"
#include "recurrent.h"
#include "BinAddDataGen.h"
#include <bitset>


int main() {
	srand((unsigned int)time(NULL));

	int64Generator gen;
	unsigned long long a1=0;
	unsigned long long a2 = 0;

	int max_len = 64;
	double** sq = km_2d::alloc(max_len, 2);
	double* label = km_1d::alloc(max_len);
	int sq_len = 0;
	Many2Many model;
	model.nn = new StackBundle(n_Layer(10), 2, 1);
	model.nn->layer[0] = new LSTM(2, 12);
	model.nn->layer[1] = new DenseLayer(12, 16, true);
	model.nn->layer[2] = new ReLU(16);
	model.nn->layer[3] = new LSTM(16, 16);
	model.nn->layer[4] = new DenseLayer(16, 16, true);
	model.nn->layer[5] = new ReLU(16);
	model.nn->layer[6] = new LSTM(16, 8);
	model.nn->layer[7] = new DenseLayer(8, 6, true);
	model.nn->layer[8] = new ReLU(6);
	model.nn->layer[9] = new VRU(6, 1, string("Sigmoid"));

 	model.nn->publish(); model.nn->allocMemory(max_len );



	optimizer optim(model.nn);
	optim.setLearingRate(0.0005);
	double** out = km_2d::alloc(max_len, 1);
	double** loss_grad = km_2d::alloc(max_len, 1);


	for (int i = 0; i < 30000; i++) {
		gen.Generate(sq, label, sq_len);

		model.FPTT(sq, out, sq_len);

		cout<< i<<"/30000 : " <<km_2d::BCEloss(loss_grad, out, label, sq_len)<<endl;
		optim.zero_grad();
		model.BPTT(loss_grad);
		optim.step();
	}
	while (true) {
		cout << "insert two positive integers:" << endl;
		cout << "one:(negative->exit)";
		cin >> a1;
		if (a1 < 0) {
			break;
		}
		cout << "two:";
		cin >> a2;
		gen.trans2bits(sq, sq_len, a1, a2);

		model.FPTT(sq, out, sq_len+1);
		cout << " model's predict:" << gen.modelpredict2digit(out, sq_len) << endl;
		cout << endl;
	}

}