/*assemble this to basic_backend.dll via:
gcc -c basic_backend.c
gcc -shared -o basic_backend.dll basic_backend.o
*/

/*DETAILS
Fairly naive ('Basic') implementations for necessary operations involving vectors and matrices
All non-transposed vectors are column vectors
KEEP ALL FUNCTIONS INDEPENDENT OF ONE ANOTHER.
/*
TODO: SPARSE MATRICES.... make a new library sparse_basic_backend
*/
#include <stdlib.h>/*malloc,calloc,realloc,free*/
#include <stdbool.h>/*for booleans*/
#include <stddef.h>/*for size_t, NULL*/
#include <string.h>/*for memcpy*/
#include <math.h>

typedef double TYPE; /*abstract away type name so can change to double precision easily*/
typedef TYPE (*func_ptr)(TYPE);
typedef struct {
  func_ptr func;
  func_ptr derivative;
} function;

__declspec(dllexport)
void * _malloc(size_t size){
  return malloc(size);
}

__declspec(dllexport)
void * _calloc(size_t num, size_t size){
  return calloc(num, size);
}

__declspec(dllexport)
void * _realloc( void *ptr, size_t new_size ){
  return realloc(ptr, new_size );
}

__declspec(dllexport)
void _free(void * ptr){
  free(ptr);
}

__declspec(dllexport)
void * _memcpy(void *str1, const void *str2, size_t n){
  memcpy(str1, str2, n);
}

__declspec(dllexport)
/*sums all elements of a vector*/
TYPE Vector_Sum(TYPE u[] ,size_t len){/*OK*/
  TYPE result = 0;
  for (size_t i = 0; i < len; i++){
      result += u[i];
  }
  return result;
}

__declspec(dllexport)
/*dot product*/
TYPE Vector_Vector(TYPE u[], TYPE v[], size_t u_len){
  static TYPE result = 0;
  for (size_t i = 0; i < u_len; i++){
      result += v[i]*u[i];
  }
  return result;
}


__declspec(dllexport)
/*vector-vector addition*/
/*NOTES: relies on host to track whether retrun is column or row vector
relies on host to check compatibility
***ret is a pointer to either a single or double dimension array****/
void Vector_Vector_Add(TYPE u[], TYPE v[], bool u_trans, bool v_trans, size_t u_len, void * _ret){/*OK*/
  if(u_trans){
    if(v_trans){
      TYPE* ret=_ret;
      for(size_t i = 0; i < u_len; i++){
          ret[i]=v[i]+u[i];
      }
    }else{
      TYPE (*ret)[u_len]=_ret;
      for(size_t i = 0; i < u_len; i++){
        for(size_t j = 0; j < u_len; j++){
          ret[i][j]=u[j]+v[i];
        }
      }
    }
  }else{
    if(v_trans){
      TYPE (*ret)[u_len]=_ret;
      for(size_t i = 0; i < u_len; i++){
        for(size_t j = 0; j < u_len; j++){
          ret[i][j]=u[i]+v[j];
        }
      }
    }else{
      TYPE* ret=_ret;
      for(size_t i = 0; i < u_len; i++){
          ret[i]=v[i]+u[i];
      }
    }
  }
}

__declspec(dllexport)
/*vector-vector subtraction*/
/*NOTES: relies on host to track whether retrun is column or row vector
relies on host to check compatibility
***ret is a pointer to either a single or double dimension array****/
void Vector_Vector_Subtract(TYPE u[], TYPE v[], bool u_trans, bool v_trans, size_t u_len, void * _ret){/*OK*/
  if(u_trans){
    if(v_trans){
      TYPE* ret=_ret;
      for(size_t i = 0; i < u_len; i++){
          ret[i]=u[i]-v[i];
      }
    }else{
      TYPE (*ret)[u_len]=_ret;
      for(size_t i = 0; i < u_len; i++){
        for(size_t j = 0; j < u_len; j++){
          ret[i][j]=u[j]-v[i];
        }
      }
    }
  }else{
    if(v_trans){
      TYPE (*ret)[u_len]=_ret;
      for(size_t i = 0; i < u_len; i++){
        for(size_t j = 0; j < u_len; j++){
          ret[i][j]=u[i]-v[j];
        }
      }
    }else{
      TYPE* ret=_ret;
      for(size_t i = 0; i < u_len; i++){
          ret[i]=u[i]-v[i];
      }
    }
  }
}

__declspec(dllexport)
/*vector-vector elementwise multiplication with broadcasting*/
/*NOTE: relies on host to track whether retrun is column or row vector
relies on the host to check compatibility
*/
void Vector_Vector_Elwise(TYPE u[], TYPE v[], bool u_trans, bool v_trans, size_t u_len, void* _ret){/*OK*/
  if(u_trans){
    if(v_trans){
      TYPE* ret=_ret;
      for (size_t i = 0; i < u_len; i++){
          ret[i]=v[i]*u[i];
      }
    }else{
      TYPE (*ret)[u_len]=_ret;
      for (size_t i = 0; i < u_len; i++){
        for (size_t j = 0; j < u_len; j++){
          ret[i][j]=u[j]*v[i];
        }
      }
    }
  }else{
    if(v_trans){
      TYPE (*ret)[u_len]=_ret;
      for (size_t i = 0; i < u_len; i++){
        for (size_t j = 0; j < u_len; j++){
          ret[i][j]=u[i]*v[j];
        }
      }
    }else{
      TYPE* ret=_ret;
      for (size_t i = 0; i < u_len; i++){
          ret[i]=v[i]*u[i];
      }
    }
  }
}

__declspec(dllexport)
 /*square matrix-vector*/
void Square_Matrix_Vector(TYPE u[], void * _m2, bool m2_trans, size_t size, TYPE ret[]){/*OK*/
  TYPE (*m2)[size]= _m2;
  if(m2_trans){
    for (size_t i = 0; i < size; i++){
      for (size_t j = 0; j < size; j++){
        ret[i]+= u[j]*m2[j][i];
      }
    }
  }else{
    for (size_t i = 0; i < size; i++){
      for (size_t j = 0; j < size; j++){
        ret[i] += u[j]*m2[i][j];
      }
    }
  }
}

__declspec(dllexport)
 /*vector-matrix elementwise addition with broadcasting*/
void Vector_Square_Matrix_Add(TYPE u[], void * _m2, bool u_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m2)[size]= _m2;
  if(u_trans){
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[j]+m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[j]+m2[i][j];
        }
      }
    }
  }else{
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[i]+m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[i]+m2[i][j];
        }
      }
    }
  }
}

__declspec(dllexport)
 /*vector-matrix elementwise multiplication with broadcasting*/
void Vector_Square_Matrix_Elwise(TYPE u[], void * _m2, bool u_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m2)[size]= _m2;
  if(u_trans){
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[j]*m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[j]*m2[i][j];
        }
      }
    }
  }else{
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[i]*m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = u[i]*m2[i][j];
        }
      }
    }
  }
}

__declspec(dllexport)
 /*matrix-matrix multiplication for two square matrices*/
void Square_Matrix_Matrix(void * _m1,void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m1)[size]= _m1;
  TYPE (*m2)[size]= _m2;
  if(m1_trans){
    if(m2_trans){
      TYPE sum=0;
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
            for (size_t k = 0; k < size; k++){
              sum = sum + m1[k][i]*m2[j][k];
            }
            ret[i][j] = sum;
            sum = 0;
          }
        }
      }else{
        TYPE sum=0;
        for (size_t i = 0; i < size; i++){
          for (size_t j = 0; j < size; j++){
            for (size_t k = 0; k < size; k++){
              sum = sum + m1[k][i]*m2[k][j];
            }
            ret[i][j] = sum;
            sum = 0;
          }
        }
    }
  }else{
    if(m2_trans){
      TYPE sum=0;
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          for (size_t k = 0; k < size; k++){
            sum = sum + m1[i][k]*m2[k][j];
          }
          ret[i][j] = sum;
          sum = 0;
        }
      }
    }else{
      TYPE sum=0;
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          for (size_t k = 0; k < size; k++){
            sum = sum + m1[i][k]*m2[j][k];
          }
          ret[i][j] = sum;
          sum = 0;
        }
      }
    }
  }
}

__declspec(dllexport)
 /*matrix-matrix elementwise multiplication for two compatible square matrices*/
void Square_Matrix_Matrix_Elwise(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m1)[size]=_m1;
  TYPE (*m2)[size]=_m2;
  if(m1_trans){
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[j][i]*m2[j][i];
        }
      }
      }else{
        for (size_t i = 0; i < size; i++){
          for (size_t j = 0; j < size; j++){
            ret[i][j] = m1[j][i]*m2[i][j];
          }
        }
    }
  }else{
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]*m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]*m2[i][j];
        }
      }
    }
  }
}

__declspec(dllexport)
 /*matrix-matrix elementwise addition for two compatible square matrices*/
void Square_Matrix_Matrix_Add(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m1)[size]=_m1;
  TYPE (*m2)[size]=_m2;
  if(m1_trans){
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[j][i]+m2[j][i];
        }
      }
      }else{
        for (size_t i = 0; i < size; i++){
          for (size_t j = 0; j < size; j++){
            ret[i][j] = m1[j][i]+m2[i][j];
          }
        }
    }
  }else{
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]+m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]+m2[i][j];
        }
      }
    }
  }
}

__declspec(dllexport)
 /*matrix-matrix elementwise subtraction for two compatible square matrices*/
void Square_Matrix_Matrix_Subtract(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret){
  TYPE (*ret)[size]=_ret;
  TYPE (*m1)[size]=_m1;
  TYPE (*m2)[size]=_m2;
  if(m1_trans){
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[j][i]-m2[j][i];
        }
      }
      }else{
        for (size_t i = 0; i < size; i++){
          for (size_t j = 0; j < size; j++){
            ret[i][j] = m1[j][i]-m2[i][j];
          }
        }
    }
  }else{
    if(m2_trans){
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]-m2[j][i];
        }
      }
    }else{
      for (size_t i = 0; i < size; i++){
        for (size_t j = 0; j < size; j++){
          ret[i][j] = m1[i][j]-m2[i][j];
        }
      }
    }
  }
}

int min_size(int a, int b){
  if(a>b){return b;}
  return a;
}

int max_size(int a, int b){
  if(a>b){return a;}
  return b;
}

__declspec(dllexport)
 /*calculates the Frobenius norm between two matrices with potentially different sizes from upperleft corner.*/
 /*does not consider transpose*/
TYPE Frobenius(void * _m1, void * _m2, size_t size_m1, size_t size_m2){
  size_t size_min = min_size(size_m1, size_m2);
  size_t size_max = max_size(size_m1, size_m2);
  TYPE (*m1)[size_m1]=_m1;
  TYPE (*m2)[size_m2]=_m2;
  TYPE (*m_max)[size_max];
  if(size_m2==size_max){m_max=m2;}
  else{m_max=m1;}
  TYPE (*ret1)[size_max]=malloc(size_max*size_max*sizeof(TYPE));
  for (size_t i = 0; i < size_min; i++){
    for (size_t j = 0; j < size_min; j++){
      ret1[i][j] = m1[i][j]-m2[i][j];
    }
  }
  for (size_t i = size_min; i < size_max; i++){
    for (size_t j = size_min; j < size_max; j++){
      ret1[i][j] = m_max[i][j];
    }
  }
  TYPE (*ret2)[size_max]=malloc(size_max*size_max*sizeof(TYPE));
  Square_Matrix_Matrix(ret1, ret1, 0, 0, size_max, ret2);
  free(ret1);
  TYPE ret;
  for(size_t i=0; i<size_max; i++){
    ret+=ret2[i][i];
  }
  return sqrt(ret);
}

__declspec(dllexport)
/*application of function vector over vector
derivative applies the derivative (should be second element of function type)
start is where respective functions start getting applied i.e. if start=3 then F[3].func will be applied to v[3] (derivative=false)
end is the last element it will be applied to
*/
void Vector_Function(function F[], TYPE v[], size_t size, size_t start, size_t end, TYPE ret[], bool derivative){
  if(derivative){
    for(size_t i=start; i<=end; i++){
      ret[i]=F[i].derivative(v[i]);
    }
  }else{
    for(size_t i=start; i<=end; i++){
      ret[i]=F[i].func(v[i]);
    }
  }
}



__declspec(dllexport)
/*checks a square matrix and makes a new matrix with 1's where the original has nonzero values*/
void Square_dA(void * _m, size_t size, void * _ret){
  TYPE (*m)[size]=_m;
  TYPE (*ret)[size]=ret;
  for(size_t i=size; i<size; i++){
    for(size_t j=size; j<size; j++){
      ret[i][j]=m[i][j]!=0;
    }
  }
}

__declspec(dllexport)
/*sum over rows or columns of a matrix
third argument (dir) is 0 for rows 1 for columns
relies on host to consider transpose
Main purpose: compute A^t * [1] where [1] is a vector of 1's whose length is A's number of columns so this denotes a sum over columns
*/

void Sum_Dir(void * _m, size_t nrows, size_t ncols, bool dir, TYPE ret[]){
  TYPE (*m)[nrows]=_m;
  if(!dir){
    for(size_t j=0; j<nrows;j++){
      for(size_t i=0; i<ncols;i++){
        ret[j]+=m[i][j];
      }
    }
  }else{
    for(size_t i=0; i<nrows;i++){
      for(size_t j=0; j<ncols;j++){
        ret[j]+=m[i][j];
      }
    }
  }
}
