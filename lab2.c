#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "omp.h"
#include "wctimer.h"

double normf(double *array, int size)
{
  int row_index;
  double sum = 0;
#pragma omp parallel for private(row_index) reduction(+ \
                                                      : sum)
  for (int i = 0; i < size; i++)
  {
    row_index = i * size;
    for (int j = 0; j < size; j++)
    {
      sum += array[row_index + j] * array[row_index + j];
    }
  }
  return sqrt(sum);
}

void init2d(double **ppter, int size, double base_value)
{
  // initialize array
  *ppter = (double *)malloc(size * size * sizeof(double));

  // initialize values
  int row_index;
#pragma omp parallel for private(row_index)
  for (int i = 0; i < size; ++i)
  {
    row_index = i * size;
    for (int j = 0; j < size; ++j)
    {
      (*ppter)[(row_index + j)] = base_value * (i + (0.1 * j));
    }
  }
}

void resetMatrix(double *array, int size)
{
  int row_index;
#pragma omp parallel for private(row_index)
  for (int i = 0; i < size; i++)
  {
    row_index = i * size;
    for (int j = 0; j < size; j++)
    {
      array[row_index + j] = 0;
    }
  }
}

void mxm_ijk(double *array1, double *array2, double *array3, int size)
{
  resetMatrix(array3, size);
  int row_index;
// compute the matrix multiplication
#pragma omp parallel for private(row_index)
  for (int i = 0; i < size; i++)
  {
    row_index = i * size;
    for (int j = 0; j < size; j++)
    {
      for (int k = 0; k < size; k++)
      {
        array3[row_index + j] += array1[row_index + k] * array2[k * size + j];
      }
    }
  }
}

void mxm_ikj(double *array1, double *array2, double *array3, int size)
{
  resetMatrix(array3, size);
  int row_index;
// compute the matrix multiplication
#pragma omp parallel for private(row_index)
  for (int i = 0; i < size; i++)
  {
    row_index = i * size;
    for (int k = 0; k < size; k++)
    {
      for (int j = 0; j < size; j++)
      {
        array3[row_index + j] += array1[row_index + k] * array2[k * size + j];
      }
    }
  }
}

void mxm_jki(double *array1, double *array2, double *array3, int size)
{
  resetMatrix(array3, size);
  int row_index;
// compute the matrix multiplication
#pragma omp parallel for private(row_index)
  for (int j = 0; j < size; j++)
  {
    for (int k = 0; k < size; k++)
    {
      row_index = k * size;
      for (int i = 0; i < size; i++)
      {
        array3[i * size + j] += array1[i * size + k] * array2[row_index + j];
      }
    }
  }
}

void mxm2(double *array1, double *array2, double *array3, int size)
{
  resetMatrix(array3, size);
  int row_index;
// compute the matrix multiplication
#pragma omp parallel for private(row_index)
  for (int i = 0; i < size; i++)
  {
    row_index = i * size;
    for (int j = 0; j < size; j++)
    {
      for (int k = 0; k < size; k++)
      {
        array3[row_index + j] += array1[row_index + k] * array2[j * size + k];
      }
    }
  }
}

void swap(double *elem1, double *elem2)
{
  double temp = *elem1;
  *elem1 = *elem2;
  *elem2 = temp;
}

void matrixTranspose(double *matrix, int subSize, int size)
{
  // cache-oblivious algorithm
  int row_index;

  if (subSize <= 32)
  {
#pragma omp parallel for private(row_index)
    for (int i = 0; i < subSize; i++)
    {
      row_index = i * size;
      for (int j = 0; j < i; j++)
      {
        swap(matrix + row_index + j, matrix + j * size + i);
      }
    }
  }
  else
  {
    int k = subSize / 2;
    matrixTranspose(matrix, k, size);
    matrixTranspose(matrix + k, k, size);
    matrixTranspose(matrix + k * size, k, size);
    matrixTranspose(matrix + k * size + k, k, size);

    int row_k = k * size;
#pragma omp parallel for private(row_index)
    for (int i = 0; i < k; i++)
    {
      row_index = i * size;
      for (int j = 0; j < k; j++)
      {
        swap(matrix + row_index + (j + k), matrix + row_index + row_k + j);
      }
    }

    if (subSize & 1)
    {
      int row_subSize = (subSize - 1) * size;
#pragma omp parallel for
      for (int i = 0; i < subSize - 1; i++)
      {
        swap(matrix + i * size + subSize - 1, matrix + row_subSize + i);
      }
    }
  }
}

void mmT(double *array, int size) { matrixTranspose(array, size, size); }

int main(int argc, char **argv)
{
  wc_timer_t t;

  if (argc < 2)
  {
    printf("usage: lab2 <size> <nthreads>\n\t<size>\t size of matrices and vectors\n");
    printf("\t<nthreads>\t number of OpenMP threads to use\n");
    exit(-1);
  }

  double times[4];
  int size;
  switch (argc)
  {
  case 2:
    size = atoi(argv[1]);
    omp_set_num_threads(1);
    break;

  case 3:
    size = atoi(argv[1]);
    omp_set_num_threads(atoi(argv[2]));
    break;
  }

  printf("size: %d, threads: %d\n", size, omp_get_num_threads());

  wc_tsc_calibrate(); // we must always calibrate the TSC timer in every program

  // initialize
  WC_INIT_TIMER(t);  // timers must be initialized before use
  WC_START_TIMER(t); // start as close to the code you want to measure

  double *array1, *array2, *array3;
  init2d(&array1, size, 1);
  init2d(&array2, size, 2);
  init2d(&array3, size, 0);

  WC_STOP_TIMER(t); // stop timer when you're done
  printf("Initialization: %10.3f ms\n\n", WC_READ_TIMER_MSEC(t));

  // ijk block
  WC_INIT_TIMER(t);  // timers must be initialized before use
  WC_START_TIMER(t); // start as close to the code you want to measure

  mxm_ijk(array1, array2, array3, size);

  WC_STOP_TIMER(t); // stop timer when you're done
  printf("\tNorm of IJK order: %10.3f\n", normf(array3, size));
  printf("IJK order: %10.3f ms\n\n", WC_READ_TIMER_MSEC(t));
  times[0] = WC_READ_TIMER_MSEC(t);

  // ikj block
  WC_INIT_TIMER(t);  // timers must be initialized before use
  WC_START_TIMER(t); // start as close to the code you want to measure

  mxm_ikj(array1, array2, array3, size);

  WC_STOP_TIMER(t); // stop timer when you're done
  printf("\tNorm of IKJ order: %10.3f\n", normf(array3, size));
  printf("IKJ order: %10.3f ms\n\n", WC_READ_TIMER_MSEC(t));
  times[1] = WC_READ_TIMER_MSEC(t);

  // jki block
  WC_INIT_TIMER(t);  // timers must be initialized before use
  WC_START_TIMER(t); // start as close to the code you want to measure

  mxm_jki(array1, array2, array3, size);

  WC_STOP_TIMER(t); // stop timer when you're done
  printf("\tNorm of JKI order: %10.3f\n", normf(array3, size));
  printf("JKI order: %10.3f ms\n\n", WC_READ_TIMER_MSEC(t));
  times[2] = WC_READ_TIMER_MSEC(t);

  // mxm2 block
  WC_INIT_TIMER(t);  // timers must be initialized before use
  WC_START_TIMER(t); // start as close to the code you want to measure

  mmT(array2, size);
  mxm2(array1, array2, array3, size);

  WC_STOP_TIMER(t); // stop timer when you're done
  printf("\tNorm of mxm2: %10.3f\n", normf(array3, size));
  printf("mxm2: %10.3f ms\n\n", WC_READ_TIMER_MSEC(t));
  times[3] = WC_READ_TIMER_MSEC(t);

  printf("times: ");
  for (int i = 0; i < 4; i++)
  {
    printf("%10.3f,", times[i]);
  }
  printf("\n");
}