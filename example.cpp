
#include "lbfgs.h"
#include <vector>
#include <iostream>

using std::vector;
using std::cout;

double func2(double * x, double * g, size_t N)
{
  size_t i;
  double dot = 0.;
  for (i=0; i<N; ++i) {
    dot += x[i] * x[i];
    g[i] = 2. * x[i];
  }
  return dot;
}

int main(){
  int N = 3;
  int M = 4;
  vector<double> x0(N);

  x0[0] = .13;
  x0[1] = 1.23;
  x0[2] = -5.03;

  LBFGS_ns::LBFGS lbfgs(&func2, &x0[0], N, M);
  lbfgs.set_max_iter(30);
  lbfgs.set_iprint(1);
  lbfgs.run();
  
  double const * x = lbfgs.get_x();
  cout << "final result:\n";
  cout << "x: " << x[0] << " " << x[1] << " " << x[2] << "\n";
  cout << "f: " << lbfgs.get_f() << "\n";
  cout << "rms: " << lbfgs.get_rms() << "\n";
  cout << "nfev: " << lbfgs.get_nfev() << "\n";
  cout << "success: " << lbfgs.success() << "\n";

}
