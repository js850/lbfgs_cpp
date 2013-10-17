#ifndef __LBFGS_CPP__
#define __LBFGS_CPP__
/**
 * an implementation of the LBFGS optimization algorithm in c++.  This
 * implemenation uses a backtracking linesearch.
 */
#include <vector>

using std::vector;
namespace LBFGS_ns{
  class LBFGS{
    private : 
      // input parameters
      /**
       * A pointer to the function that computes the function and gradient
       */
      double (* func_f_grad_)(double *, double *, int);

      int N_; /**< The number of elements in the search space */
      int M_; /**< The lenth of the LBFGS memory */
      double tol_; /**< The tolerance for the rms gradient */
      double maxstep_; /**< The maximum step size */
      double max_f_rise_; /**< The maximum the function is allowed to rise in a
                           * given step.  This is the criterion for the
                           * backtracking line search.
                           */
      int maxiter_; /**< The maximum number of iterations */
      int iprint_;

      int iter_number_; /**< The current iteration number */
      int nfev_; /**< The number of function evaluations */

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
      int k_; /**< Counter for how many times the memory has been updated */

      // 
      std::vector<double> step_;

    public :
      /**
       * Constructor
       */
      LBFGS(
          double (*func)(double *, double *, int), 
          double const * x0, 
          int N, 
          int M);
          //double tol,
          //double maxstep,
          //double max_f_rise,
          //double H0,
          //int maxiter
          //);

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

      // functions for setting the parameters
      void set_H0(double);
      void set_tol(double tol) { tol_ = tol; }
      void set_maxstep(double maxstep) { maxstep_ = maxstep; }
      void set_max_f_rise(double max_f_rise) { max_f_rise_ = max_f_rise; }
      void set_max_iter(int max_iter) { maxiter_ = max_iter; }
      void set_iprint(int iprint) { iprint_ = iprint; }

      // functions for accessing the results
      double const * get_x() { return &x_[0]; }
      double const * get_g() { return &g_[0]; }
      double get_f() { return f_; }
      double get_rms() { return rms_; }
      int get_nfev() { return nfev_; }
      int get_niter() { return iter_number_; }
      int success() { return stop_criterion_satisfied(); }

    private :

      /**
       * Add a step to the LBFGS Memory
       * This updates s_, y_, rho_, H0_, and k_
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

#endif
