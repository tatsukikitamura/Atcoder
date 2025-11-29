// cl cmd.cpp -O2 /std:c++17
#include <iostream>
#include <algorithm>
#include <bitset>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <queue>
#include <deque>
#include <set>
#include <stack>
#include <string>
#include <cstring>
#include <utility>
#include <vector>
#include <complex>
#include <valarray>
#include <fstream>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <climits>
#include <random>
#include <time.h>

#define rep(i,n) for(int i=0;i<n;i++)
#define rrep(i,n) for(int i=(n-1);i>=0;i--)
using namespace std;
typedef long long ll;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vl;
typedef vector<bool> vb;
typedef vector<double> vd;

bool debug = false; // false : 提出, true : fin - fout
double time_ratio = 1.0;
string file_No = "0000";
ifstream fin;
ofstream fout;
ofstream fout_debug;
ofstream fout_score;
int Gaussian_size = 1000000;
vector<double> Gaussian_(Gaussian_size);
struct timespec timeini;

const int InfL = 2000000000;
const ll InfLL = 4000000000000000000LL;

vd Dinv;
vd Sqrt;

void debug_init() {
	string txt = ".txt";

	string ifname = "in\\";
	ifname = ifname + file_No + txt;
	fin.open(ifname);
	string ofname = "out\\";
	ofname = ofname + file_No + txt;
	fout.open(ofname);
	string ofname_debug = "debug\\";
	ofname_debug = ofname_debug + file_No + txt;
	fout_debug.open(ofname_debug);
	string ofname_score = "score\\";
	ofname_score = ofname_score + file_No + txt;
	fout_score.open(ofname_score);

	time_ratio = 1.15; // 14900K
	//time_ratio = 0.88; // 5950X
	//time_ratio = 0.7; // 1255U
}

void debug_end() {
	fin.close();
	fout.close();
	fout_debug.close();
}

double time_diff()
{
	struct timespec timetmp;
	timespec_get(&timetmp, TIME_UTC);
	double ans = timetmp.tv_nsec - timeini.tv_nsec;
	ans += (double)(timetmp.tv_sec - timeini.tv_sec) * 1000000000.0;
	return time_ratio * ans / 1000000.0;
}

int xor128() {
	static int x = 123456789, y = 362436069, z = 521288629, w = 88675123;
	int t = (x ^ (x << 11));
	x = y; y = z; z = w;
	return (w = (w ^ (w >> 19)) ^ (t ^ (t >> 8)));
}

double RAND_() {
	double rand_ = (double)(xor128()) / (double)INT32_MAX;
	return rand_;
}

double Gaussian() {
	double X = RAND_();
	double Y = RAND_();
	double Z = sqrt(-2.0 * log(X)) * cos(2.0 * acos(-1.0) * Y);
	return Z;
}

int distXY(int xfrom, int yfrom, int xto, int yto) {
	int xdiff = abs(xfrom - xto);
	if (xdiff > 5000) xdiff = 10000 - xdiff;
	int ydiff = abs(yfrom - yto);
	if (ydiff > 5000) ydiff = 10000 - ydiff;
	return xdiff * xdiff + ydiff * ydiff;
}


int N, T, M, K, L;
class WeightedRandomIndex {
public:
	vd prefix;
	double total = 0.0;
	void reset(vd& weights) {
		int n = weights.size();
		prefix.resize(n + 1);
		prefix[0] = 0.0;
		rep(i, n) {
			double w = weights[i];
			prefix[i + 1] = prefix[i] + w;
		}
		total = prefix.back();
	}
	int sample() {
		double r = RAND_() * total;
		auto it = upper_bound(prefix.begin() + 1, prefix.end(), r);
		int idx = (it - prefix.begin() - 1);
		return idx;
	}
};

WeightedRandomIndex weightedRandomIndex;

struct Input {
	vector<vd> X, Y;
	vd Vx, Vy;
	void readProblem(istream& in = cin) {
		in >> N >> T >> M >> K >> L;
		X.resize(N, vd(T + 10));
		Y.resize(N, vd(T + 10));
		Vx.resize(N);
		Vy.resize(N);
		rep(n, N) {
			in >> X[n][0] >> Y[n][0] >> Vx[n] >> Vy[n];
			rep(t, T + 9) {
				X[n][t + 1] = X[n][t] + Vx[n];
				if (X[n][t + 1] < 0.0) X[n][t + 1] += L;
				if (X[n][t + 1] > L) X[n][t + 1] -= L;
				Y[n][t + 1] = Y[n][t] + Vy[n];
				if (Y[n][t + 1] < 0.0) Y[n][t + 1] += L;
				if (Y[n][t + 1] > L) Y[n][t + 1] -= L;
			}
		}

		return;
	}
};

Input input;

struct Output {
	vi NtoM;
	vi NtoK;
	vi NtoNfrom;
	vi NtoNfromNew;
	vd NtoScore;
	vd NtoScoreNew;
	vector<vector<pii>> MKtoTN;
	vector<vector<pii>> MKtoTNold;
	vd scOld;
	vd scNew;

	vi Xdiff, Ydiff;

	void resize_all() {
		NtoM.resize(N);
		NtoK.resize(N);
		NtoScore.resize(N);
		NtoScoreNew.resize(N);
		NtoNfrom.resize(N, -1);
		NtoNfromNew.resize(N, -1);
		MKtoTN.resize(M, vector<pii>(K));
		MKtoTNold.resize(M, vector<pii>(K));
		Xdiff.resize(K);
		Ydiff.resize(K);
		scOld.resize(M);
		scNew.resize(M);
		rep(m, M) {
			rep(k, K) {
				int n = m * K + k;
				NtoM[n] = m;
				NtoK[n] = k;
				if (k > 0)
					NtoNfrom[n] = MKtoTN[m][k - 1].second;
				MKtoTN[m][k] = { k * 30, n };
			}
		}
	}
	void writeSolution(ostream& out = cout) {
		rep(n, N) {
			if (NtoNfrom[n] == -1) continue;
			int m = NtoM[n];
			int k = NtoK[n];
			int t = MKtoTN[m][k].first;
			int nfrom = NtoNfrom[n];
			out << t << " " << n << " " << nfrom << "\n";
		}

		return;
	}

	void Memory(int m) {
		rep(k, K)
			MKtoTNold[m][k] = MKtoTN[m][k];
	}
	void Msort(int m) {
		sort(MKtoTN[m].begin(), MKtoTN[m].end());
	}

	double EvalTmp(int m, int Kd, double scth = 1e20) {
		double sc = 0.0;
		double Xc = 0.0, Yc = 0.0;
		double Vx = 0.0, Vy = 0.0;

		rep(k, K) {
			int t = MKtoTN[m][k].first;
			int n = MKtoTN[m][k].second;
			int xNext = input.X[n][t];
			int yNext = input.Y[n][t];
			int vxNext = input.Vx[n];
			int vyNext = input.Vy[n];
			if (k == 0) {
				Xc = xNext;
				Yc = yNext;
				Vx = vxNext;
				Vy = vyNext;
				Xdiff[0] = 0;
				Ydiff[0] = 0;
				NtoNfromNew[n] = -1;
				NtoScoreNew[n] = 0;
			}
			else {
				int tDiff = t - MKtoTN[m][k - 1].first;
				Xc += Vx * tDiff;
				Yc += Vy * tDiff;
				int distmin = InfL;
				double kmin = -1;
				int xNextDiff = xNext - (int)(Xc + 0.5);
				int yNextDiff = yNext - (int)(Yc + 0.5);
				xNextDiff /= 10;
				yNextDiff /= 10;
				rep(kk, k) {
					int disttmp = distXY(Xdiff[kk], Ydiff[kk], xNextDiff, yNextDiff);
					if (distmin > disttmp) {
						distmin = disttmp;
						kmin = kk;
					}
				}
				double scTmp = Sqrt[distmin / 100] * 100.0;
				NtoScoreNew[n] = (double)distmin * Sqrt[k];
				sc += scTmp;
				int krem = max(Kd + K - 1 - k - 10, 0);
				if (sc + 1000.0 * krem > scth) return InfLL;
				Xdiff[k] = xNextDiff;
				Ydiff[k] = yNextDiff;
				Vx = ((double)k * Vx + vxNext) * Dinv[k + 1];
				Vy = ((double)k * Vy + vyNext) * Dinv[k + 1];
				NtoNfromNew[n] = MKtoTN[m][kmin].second;
			}
		}
		scNew[m] = sc;
		return sc;
	}

	void Accept(int m) {
		rep(k, K) {
			int n = MKtoTN[m][k].second;
			NtoNfrom[n] = NtoNfromNew[n];
			NtoScore[n] = NtoScoreNew[n];
			NtoM[n] = m;
			NtoK[n] = k;
		}
		scOld[m] = scNew[m];
	}

	void Rollback(int m) {
		rep(k, K)
			MKtoTN[m][k] = MKtoTNold[m][k];
	}

	double EvalAll() {
		double sc = 0.0;
		rep(m, M) {
			sc += EvalTmp(m, 0);
			Accept(m);
		}
		return sc;
	}
};

Output output;

void solve_init() {
	output.resize_all();
	rep(i, Gaussian_size)
		Gaussian_[i] = Gaussian();
	Dinv.resize(100);
	rep(i, 99)
		Dinv[i + 1] = 1.0 / (double)(i + 1);
	Sqrt.resize(1000000);
	rep(i, 1000000)
		Sqrt[i] = sqrt(i);

	return;
}

void solveSA() {
	double sc = output.EvalAll();
	weightedRandomIndex.reset(output.NtoScore);

	double start_temp = 6000.0;
	double end_temp = 1.0;
	double TIME_INIT = time_diff();
	double TIME_LIMIT = 1985.0 - TIME_INIT;
	ll loop_count = 0;
	double time_now = 0.0;
	double temp = 0.0;

	int mfrom = -1, mto = -1;
	int kfrom = -1, kto = -1;
	int nfrom = -1, nto = -1;
	int tfrom = -1, tto = -1;

	while (1) {
		//break;
		if (loop_count % 1000 == 0) {
			time_now = time_diff() - TIME_INIT;
			if (time_now > TIME_LIMIT)
				break;
			double progress = time_now / TIME_LIMIT;
			progress = pow(progress, 0.7);
			temp = start_temp + (end_temp - start_temp) * progress;
		}

		loop_count++;
		double scdiff = 0;

		double scth = -temp * log(RAND_());
		//scth = 0.0;
		int seed = xor128() % 2;
		int type = 0;
		if (seed < 1)
			type = 0;
		else if (seed < InfL)
			type = 1;
		if (type == 0) {
			//continue;
			nfrom = weightedRandomIndex.sample();
			mfrom = output.NtoM[nfrom];
			kfrom = output.NtoK[nfrom];
			if (xor128() % K == 0) {
				mfrom = xor128() % M;
				kfrom = 0;
				nfrom = output.MKtoTN[mfrom][kfrom].second;
			}

			tfrom = output.MKtoTN[mfrom][kfrom].first;
			tfrom += (int)(Gaussian_[xor128() % Gaussian_size] * 150.0 + 0.5);
			if (tfrom < 0) continue;
			if (tfrom >= T) continue;
			scdiff -= output.scOld[mfrom];
			output.Memory(mfrom);
			output.MKtoTN[mfrom][kfrom].first = tfrom;
			output.Msort(mfrom);
			scdiff += output.EvalTmp(mfrom, 0, scth - scdiff);
		}
		else if (type == 1) {
			//continue;
			nfrom = weightedRandomIndex.sample();
			mfrom = output.NtoM[nfrom];
			kfrom = output.NtoK[nfrom];
			if (xor128() % K == 0) {
				mfrom = xor128() % M;
				kfrom = 0;
				nfrom = output.MKtoTN[mfrom][kfrom].second;
			}

			tfrom = output.MKtoTN[mfrom][kfrom].first;
			nto = weightedRandomIndex.sample();
			mto = output.NtoM[nto];
			kto = output.NtoK[nto];
			if (xor128() % K == 0) {
				mto = xor128() % M;
				kto = 0;
				nto = output.MKtoTN[mto][kto].second;
			}

			tto = output.MKtoTN[mto][kto].first;
			if (mfrom == mto) continue;

			scdiff -= output.scOld[mfrom];
			scdiff -= output.scOld[mto];
			output.Memory(mfrom);
			output.Memory(mto);
			output.MKtoTN[mfrom][kfrom].second = nto;
			output.MKtoTN[mto][kto].second = nfrom;
			output.Msort(mfrom);
			output.Msort(mto);
			scdiff += output.EvalTmp(mfrom, K, scth - scdiff);
			scdiff += output.EvalTmp(mto, 0, scth - scdiff);
		}

		if (scdiff < scth) { 
			sc += scdiff;
			//if (scdiff < 0) cout << type << " " << sc << endl;

			weightedRandomIndex.reset(output.NtoScore);
			if (type == 0) {
				output.Accept(mfrom);
			}
			else if (type == 1) {
				output.Accept(mfrom);
				output.Accept(mto);
			}
		}
		else {
			if (type == 0) {
				output.Rollback(mfrom);

			}
			else if (type == 1) {
				output.Rollback(mfrom);
				output.Rollback(mto);
			}
		}
	}
	return;
}

int main(int argc, char* argv[]) {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);
	if (argc >= 2) file_No = argv[1];
	timespec_get(&timeini, TIME_UTC);
	if (debug)
		debug_init();
	debug ? input.readProblem(fin) : input.readProblem();
	solve_init();
	solveSA();
	debug ? output.writeSolution(fout) : output.writeSolution();
	if (debug)
		debug_end();
	return 0;
}
