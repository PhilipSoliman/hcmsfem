#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <vector>

// extern "C"
// {
//    // CG: main function
//    bool custom_cg(double *Af, double *b, double *x, double *x_m, double *alpha, double *beta, const int size, int &niters, const int max_iter, const double tol, const bool save_residuals, double *residuals, const bool exact_convergence, double *x_exact, double *errors);

//    // test function
//    void TEST();
// }
// Helper: copy from pointer to vector
std::vector<double> ptr_to_vec(const double *ptr, int n)
{
   return std::vector<double>(ptr, ptr + n);
}

// Helper: copy from vector to pointer
void vec_to_ptr(const std::vector<double> &vec, double *ptr)
{
   std::copy(vec.begin(), vec.end(), ptr);
}

// Helper: copy from pointer to flat array to matrix (row-major)
std::vector<std::vector<double>> ptr_to_mat(const double *ptr, int n)
{
   std::vector<std::vector<double>> mat(n, std::vector<double>(n));
   for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
         mat[i][j] = ptr[i * n + j];
   return mat;
}

// Linear algebra functions using std::vector
double dot_product(const std::vector<double> &a, const std::vector<double> &b)
{
   double sum = 0.0;
   for (size_t i = 0; i < a.size(); ++i)
      sum += a[i] * b[i];
   return sum;
}

std::vector<double> matrix_vector_product(const std::vector<std::vector<double>> &A, const std::vector<double> &x)
{
   std::vector<double> prod(A.size(), 0.0);
   for (size_t i = 0; i < A.size(); ++i)
      for (size_t j = 0; j < x.size(); ++j)
         prod[i] += A[i][j] * x[j];
   return prod;
}

std::vector<double> add(const std::vector<double> &a, const std::vector<double> &b)
{
   std::vector<double> sum(a.size());
   for (size_t i = 0; i < a.size(); ++i)
      sum[i] = a[i] + b[i];
   return sum;
}

std::vector<double> subtract(const std::vector<double> &a, const std::vector<double> &b)
{
   std::vector<double> diff(a.size());
   for (size_t i = 0; i < a.size(); ++i)
      diff[i] = a[i] - b[i];
   return diff;
}

std::vector<double> scalar_multiplication(const std::vector<double> &a, double scalar)
{
   std::vector<double> prod(a.size());
   for (size_t i = 0; i < a.size(); ++i)
      prod[i] = a[i] * scalar;
   return prod;
}

// CG supporting functions
std::vector<double> initial_residual(const std::vector<std::vector<double>> &A, const std::vector<double> &b, const std::vector<double> &x)
{
   return subtract(b, matrix_vector_product(A, x));
}

double calculate_alpha(const double r_dot_r, const std::vector<double> &p, const std::vector<double> &Ap)
{
   double p_dot_Ap = dot_product(p, Ap);
   return r_dot_r / p_dot_Ap;
}

std::vector<double> calculate_solution(const std::vector<double> &x, const std::vector<double> &p, double alpha)
{
   return add(x, scalar_multiplication(p, alpha));
}

std::vector<double> calculate_residual(const std::vector<double> &r, const double alpha, const std::vector<double> &Ap)
{
   return subtract(r, scalar_multiplication(Ap, alpha));
}

double calculate_beta(const double r_dot_r, const std::vector<double> &r_m)
{
   double r_m_dot_r_m = dot_product(r_m, r_m);
   return r_m_dot_r_m / r_dot_r;
}

std::vector<double> calculate_search_direction(const std::vector<double> &p, const std::vector<double> &r, const double beta)
{
   return add(r, scalar_multiplication(p, beta));
}

// Main CG function (C interface, but uses std::vector internally)
bool custom_cg(
    double *Af,
    double *b,
    double *x,
    double *x_m,
    double *alpha,
    double *beta,
    const int size,
    int &niters,
    const int max_iter,
    const double tol,
    const bool save_residuals,
    double *residuals,
    const bool exact_convergence,
    double *x_exact,
    double *errors)
{
   auto A = ptr_to_mat(Af, size);
   std::vector<double> b_vec = ptr_to_vec(b, size);
   std::vector<double> x_vec = ptr_to_vec(x, size);
   std::vector<double> x_m_vec = ptr_to_vec(x_m, size);
   std::vector<double> x_exact_vec = x_exact ? ptr_to_vec(x_exact, size) : std::vector<double>(size, 0.0);

   bool success = false;

   std::vector<double> r(size), r_m(size), p(size), Ap(size), em(size), r_new(size);

   r = initial_residual(A, b_vec, x_vec);
   p = r;

   double r0_dot_r0 = dot_product(r, r);
   double r0_norm = std::sqrt(r0_dot_r0);

   double e0_norm = 1.0;
   if (exact_convergence && x_exact)
   {
      auto e0 = subtract(x_exact_vec, x_vec);
      e0_norm = std::sqrt(dot_product(e0, e0));
   }

   int j = 0;
   for (; j < max_iter; ++j)
   {
      double r_dot_r = dot_product(r, r);
      if (save_residuals && residuals)
      {
         residuals[j] = std::sqrt(r_dot_r);
      }

      if (exact_convergence && x_exact)
      {
         em = subtract(x_exact_vec, x_m_vec);
         double em_norm = std::sqrt(dot_product(em, em));
         errors[j] = em_norm;
         if (em_norm / e0_norm < tol)
         {
            success = true;
            break;
         }
      }
      else
      {
         if (std::sqrt(r_dot_r) / r0_norm < tol)
         {
            success = true;
            break;
         }
      }

      Ap = matrix_vector_product(A, p);
      double alpha_j = calculate_alpha(r_dot_r, p, Ap);
      alpha[j] = alpha_j;

      x_m_vec = calculate_solution(x_m_vec, p, alpha_j);
      r_m = calculate_residual(r, alpha_j, Ap);

      double beta_j = calculate_beta(r_dot_r, r_m);
      beta[j] = beta_j;

      p = calculate_search_direction(p, r_m, beta_j);

      r = r_m;
   }

   // Copy results back to output pointers
   vec_to_ptr(x_m_vec, x_m);

   niters = j;
   return success;
}

// Test function using std::vector
void TEST()
{
   printf("Hello from C++!\n");

   const int size = 4;
   std::vector<std::vector<double>> A(size, std::vector<double>(size, 0.0));
   for (int i = 0; i < size; ++i)
   {
      A[i][i] = 1.0;
      for (int j = 0; j < size; ++j)
         printf("A[%d][%d]: %f\n", i, j, A[i][j]);
   }
   std::vector<double> x(size, 0.0);
   std::vector<double> b(size, 1.0);

   for (int i = 0; i < size; ++i)
      printf("b[%d]: %f\n", i, b[i]);

   printf("Testing initial_residual function\n");
   std::vector<double> r = initial_residual(A, b, x);
   for (int i = 0; i < size; ++i)
      printf("r[%d]: %f\n", i, r[i]);

   printf("Testing calculate_alpha function\n");
   std::vector<double> p = r;
   double r_dot_r = dot_product(r, r);
   std::vector<double> Ap = matrix_vector_product(A, p);
   double alpha = calculate_alpha(r_dot_r, p, Ap);
   printf("alpha: %f\n", alpha);

   printf("Testing calculate_solution function\n");
   std::vector<double> x_m = calculate_solution(x_m, p, alpha);
   for (int i = 0; i < size; ++i)
      printf("x_m[%d]: %f\n", i, x_m[i]);

   printf("Testing calculate_residual function\n");
   std::vector<double> r_new = calculate_residual(r, alpha, Ap);
   for (int i = 0; i < size; ++i)
      printf("r_new[%d]: %f\n", i, r_new[i]);

   printf("Testing calculate_beta function\n");
   std::vector<double> r_m = r_new;
   double beta = calculate_beta(r_dot_r, r_m);
   printf("beta: %f\n", beta);

   printf("Testing search_direction_update function\n");
   p = calculate_search_direction(p, r_m, beta);
   for (int i = 0; i < size; ++i)
      printf("p[%d]: %f\n", i, p[i]);
}

// Python wrappers
static PyObject *py_TEST(PyObject *self, PyObject *args)
{
   TEST();
   Py_RETURN_NONE;
}

static PyObject *py_custom_cg(PyObject *self, PyObject *args)
{
   PyObject *Af_obj, *b_obj, *x_obj, *x_m_obj, *alpha_obj, *beta_obj, *niters_obj;
   int size, max_iter;
   double tol;
   bool save_residuals, exact_convergence;
   PyObject *residuals_obj = nullptr, *x_exact_obj = nullptr, *errors_obj = nullptr;

   // Parse arguments (adjust as needed for your API)
   if (!PyArg_ParseTuple(
           args,
           "OOOOOOiOidpOpOO",
           &Af_obj,
           &b_obj,
           &x_obj,
           &x_m_obj,
           &alpha_obj,
           &beta_obj,
           &size,
           &niters_obj,
           &max_iter,
           &tol,
           &save_residuals,
           &residuals_obj,
           &exact_convergence,
           &x_exact_obj,
           &errors_obj))
   {
      return NULL;
   }

   // Convert to NumPy arrays (double, contiguous)
   PyArrayObject *Af_arr = (PyArrayObject *)PyArray_FROM_OTF(Af_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
   if (!Af_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert Af array.");
      return NULL;
   }
   PyArrayObject *b_arr = (PyArrayObject *)PyArray_FROM_OTF(b_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
   if (!b_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert b array.");
      return NULL;
   }
   PyArrayObject *x_arr = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
   if (!x_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert x array.");
      return NULL;
   }
   PyArrayObject *x_m_arr = (PyArrayObject *)PyArray_FROM_OTF(x_m_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
   if (!x_m_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert x_m array.");
      return NULL;
   }
   PyArrayObject *alpha_arr = (PyArrayObject *)PyArray_FROM_OTF(alpha_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
   if (!alpha_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert alpha array.");
      return NULL;
   }
   PyArrayObject *beta_arr = (PyArrayObject *)PyArray_FROM_OTF(beta_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
   if (!beta_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert beta array.");
      return NULL;
   }
   PyArrayObject *niters_arr = (PyArrayObject *)PyArray_FROM_OTF(niters_obj, NPY_INT32, NPY_ARRAY_INOUT_ARRAY);
   if (!niters_arr)
   {
      PyErr_SetString(PyExc_RuntimeError, "Failed to convert niters array.");
      return NULL;
   }

   // Handle optional arrays
   PyArrayObject *residuals_arr = nullptr;
   if (residuals_obj)
   {
      residuals_arr = (PyArrayObject *)PyArray_FROM_OTF(residuals_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
      if (!residuals_arr)
      {
         PyErr_SetString(PyExc_RuntimeError, "Failed to convert residuals array.");
         return NULL;
      }
   }
   PyArrayObject *x_exact_arr = nullptr;
   if (x_exact_obj)
   {
      x_exact_arr = (PyArrayObject *)PyArray_FROM_OTF(x_exact_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
      if (!x_exact_arr)
      {
         PyErr_SetString(PyExc_RuntimeError, "Failed to convert x_exact array.");
         return NULL;
      }
   }
   PyArrayObject *errors_arr = nullptr;
   if (errors_obj)
   {
      errors_arr = (PyArrayObject *)PyArray_FROM_OTF(errors_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
      if (!errors_arr)
      {
         PyErr_SetString(PyExc_RuntimeError, "Failed to convert errors array.");
         return NULL;
      }
   }

   double *Af = (double *)PyArray_GETPTR1(Af_arr, 0);
   double *b = (double *)PyArray_GETPTR1(b_arr, 0);
   double *x = (double *)PyArray_GETPTR1(x_arr, 0);
   double *x_m = (double *)PyArray_GETPTR1(x_m_arr, 0);
   double *alpha = (double *)PyArray_GETPTR1(alpha_arr, 0);
   double *beta = (double *)PyArray_GETPTR1(beta_arr, 0);
   int *niters = (int *)PyArray_GETPTR1(niters_arr, 0);
   double *residuals = residuals_arr ? (double *)PyArray_GETPTR1(residuals_arr, 0) : nullptr;
   double *x_exact = x_exact_arr ? (double *)PyArray_GETPTR1(x_exact_arr, 0) : nullptr;
   double *errors = errors_arr ? (double *)PyArray_GETPTR1(errors_arr, 0) : nullptr;

   bool success = custom_cg(Af, b, x, x_m, alpha, beta, size, *niters, max_iter, tol,
                            save_residuals, residuals, exact_convergence, x_exact, errors);

   if (success)
      Py_RETURN_TRUE;
   else
      Py_RETURN_FALSE;
}

// Method table
static PyMethodDef CustomCGMethods[] = {
    {"custom_cg", py_custom_cg, METH_VARARGS, "Run CG in c++"},
    {"TEST", py_TEST, METH_NOARGS, "Run a C++ test function. Tests all helper functions for CG."},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef customcgmodule = {
    PyModuleDef_HEAD_INIT,
    "custom_cg", // name of module
    NULL,        // module documentation
    -1,          // size of per-interpreter state of the module
    CustomCGMethods};

// Module init function
PyMODINIT_FUNC PyInit_custom_cg(void)
{
   import_array();
   return PyModule_Create(&customcgmodule);
}
