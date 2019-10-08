#include "unit.h"
#include "optimizer.h"
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


	StackBundle* frnn = new StackBundle(n_Layer(6), 2, 16);
	frnn->layer[0] = new LSTM(2, 12);
	frnn->layer[1] = new DenseLayer(12, 16, true);
	frnn->layer[2] = new ReLU(16);
	frnn->layer[3] = new LSTM(16, 16);
	frnn->layer[4] = new DenseLayer(16, 16, true);
	frnn->layer[5] = new ReLU(16);
	frnn->publish();

	StackBundle* brnn = new StackBundle(n_Layer(6), 2, 16);
	brnn->layer[0] = new LSTM(2, 12);
	brnn->layer[1] = new DenseLayer(12, 16, true);
	brnn->layer[2] = new ReLU(16);
	brnn->layer[3] = new LSTM(16, 16);
	brnn->layer[4] = new DenseLayer(16, 16, true);
	brnn->layer[5] = new ReLU(16);
	brnn->publish();
	
	StackBundle* mnn = new StackBundle(n_Layer(4), 16, 1);
	mnn->layer[0] = new DenseLayer(16, 8);
	mnn->layer[1] = new ReLU(8);
	mnn->layer[2] = new DenseLayer(8, 1);
	mnn->layer[3] = new Sigmoid(1);

	mnn->publish();
	
	BidirectionalModel model(frnn, brnn, mnn);
	model.allocMemory(max_len);


	optimizer optim1(frnn);
	optimizer optim2(brnn);
	optimizer optim3(mnn);

	optim1.setLearingRate(0.0005);
	optim2.setLearingRate(0.0005);
	optim3.setLearingRate(0.0005);

	double** out = km_2d::alloc(max_len, 1);
	double** loss_grad = km_2d::alloc(max_len, 1);


	for (int i = 0; i < 30000; i++) {
		gen.Generate(sq, label, sq_len);
		model.FPTT(sq, out, sq_len);

		cout<< i<<"/30000 : " <<km_2d::BCEloss(loss_grad, out, label, sq_len)<<endl;
		optim1.zero_grad();
		optim2.zero_grad();
		optim3.zero_grad();
		model.BPTT(loss_grad);
		optim1.step();
		optim2.step();
		optim3.step();
	}
	while (true) {
		cout << "insert two positive integers:" << endl;
		cout << "one:(33333->exit)";
		cin >> a1;
		if (a1 == (unsigned long long)33333) {
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
