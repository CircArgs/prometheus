/*assemble this to basic_backend.dll via:
gcc -c basic_backend.c
gcc - shared -o basic_backend.dll basic_backend.o
*/

/*DETAILS
Fairly naive ('Basic') implementations for necessary operations involving vectors and matrices
All non-transposed vectors are column vectors
/*
TODO: SPARSE MATRICES.... make a new library sparse_basic_backend
*/
#include <stdlib.h>/*malloc*/
#include <stdbool.h>/*for booleans*/
#include <stddef.h>/*for size_t, NULL*/

typedef double TYPE; /*abstract away type name so can change to double precision easily*/
typedef TYPE (*func_ptr)(TYPE);

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
        ret[i] += u[i]*m2[j][i];
      }
    }
  }else{
    for (size_t i = 0; i < size; i++){
      for (size_t j = 0; j < size; j++){
        ret[i] += u[i]*m2[i][j];
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

__declspec(dllexport)
/*application of function vector over vector*/
void Vector_Function(func_ptr F[], TYPE v[], size_t size,size_t start, size_t end, TYPE ret[]){
  for(size_t i=0; i<size; i++){
    ret[i]=F[i](v[i]);
  }
}

__declspec(dllexport)
/*application of function vector over vector*/
void Vector_Function_Ovw(func_ptr F[], TYPE v[], size_t start, size_t end){
  for(size_t i=start; i<=end; i++){
    v[i]=F[i](v[i]);
  }
}
