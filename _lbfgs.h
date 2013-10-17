/**
 * an implimentation of the LBFGS optimization algorithm in c++.  This
 * implemenation uses a backtracking linesearch.
 */
#include <vector>

using std::vector;
namespace LBFGS_ns{
  class LBFGS{
    private : 
      double (* func_f_grad_)(double *, double *, int);

      int M_; /**< The lenth of the LBFGS memory */
      int N_; /**< The number of elements in the search space */
      int k_; /**< Counter for how many times the memory has been updated */
      int maxiter_; /**< The maximum number of iterations */
      double tol_; /**< The tolerance for the rms gradient */
      double maxstep_; /**< The maximum step size */
      int nfev_; /**< The number of function evaluations */
      double max_f_rise_; /**< The maximum the function is allowed to rise in a
                           * given step.  This is the criterion for the
                           * backtracking line search.
                           */
      int iter_number_; /**< The current iteration number */

      // variables representing the state of the system
      std::vector<double> x_;
      double f_;
      std::vector<double> g_;
      double rms_;

      // places to store the lbfgs memory
      std::vector<vector<double> > s_;
      std::vector<vector<double> > y_;
      std::vector<double> rho_;
      double H0_;

      // 
      std::vector<double> step_;

    public :
      /**
       * Constructor
       */
      LBFGS(double (*func)(double *, double *, int),
          double const * x0, int N, int M);

      /**
       * Destructor
       */
      ~LBFGS();

      /**
       * Do one iteration iteration of the optimization algorithm
       */
      void one_iteration();

      /**
       * Run the optimzation algorithm until the tolerance is satisfied or
       * until the maximum number ofg iterations is reached
       */
      void run();

    private :

      /**
       * Add a step to the LBFGS Memory
       * This updates s_, y_, rho_, and H0_
       */
      void update_memory(
          std::vector<double> & xold,
          std::vector<double> & gold,
          std::vector<double> & xnew,
          std::vector<double> & gnew
          );

      /**
       * Compute the LBFGS step from the memory
       */
      void compute_lbfgs_step();

      /**
       * Take the step and do a backtracking linesearch if necessary.
       * Apply the maximum step size constraint and ensure that the function
       * does not rise more than the allowed amount.
       */
      double backtracking_linesearch();

      /**
       * Return true if the termination condition is satisfied, false otherwise
       */
      int stop_criterion_satisfied();

      /**
       * Compute the energy and gradient of the objective function
       */
      void compute_func_gradient(std::vector<double> & x, double & energy,
          std::vector<double> & gradient);

  };
}
