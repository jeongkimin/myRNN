#pragma once
#include "core.h"
#include <bitset>
class int64Generator {
public:

	int64Generator() {
		srand((unsigned int)time(NULL));
	}
	unsigned long long modelpredict2digit(double**& bins, int& len) {
		unsigned long long d = 0;
		unsigned long long v(1);
		for (int i = 0; i < len; i++) {
			if (bins[i][0] > 0.5) {
				d += v;
			}
			v *= (unsigned long long)2;
		}
		return d;
	}
	void trans2bits(double**& sq, int& len, unsigned long long& d1, unsigned long long& d2) {
		km_2d::fill_zero(sq, 64, 2);
		bitset<64> b1(d1);
		bitset<64> b2(d2);
		for (int i = 63; i >= 0; i--) {
			if ((int)b1[i] == 1 || (int)b2[i] == 1) {
				len = i + 2;
				break;
			}
		}
		for (int i = 0; i < len; i++) {
			sq[i][0] = (double)b1[i];
			sq[i][1] = (double)b2[i];
		}
		sq[len-1][0] = 0.0;
		sq[len-1][1] = 0.0;
	}

	void Generate(double**& sq, double*& tar, int& len) {


		unsigned long long value1 = (unsigned long long)(rand())
			* (unsigned long long)(rand())
			* (unsigned long long)(rand());
		unsigned long long value2 = (unsigned long long)(rand())
			* (unsigned long long)(rand())
			* (unsigned long long)(rand())
			* (unsigned long long)(rand());
		bitset<64> bin1(value1);
		bitset<64> bin2(value2);
		bitset<64> bin3(value1 + value2);
		for (int i = 63; i >= 0; i--) {
			if ((int)bin3[i] == 1) {
				len = i + 1;
				break;
			}
		}
		km_2d::fill_zero(sq, 64, 2);
		km_1d::fill_zero(tar, 64);
		for (int i = 0; i < len; i++) {
			sq[i][0] = (double)bin1[i];
			sq[i][1] = (double)bin2[i];
			tar[i] = (double)bin3[i];
		}
	}
};