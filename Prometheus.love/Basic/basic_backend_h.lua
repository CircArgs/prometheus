--[[
This file needs to contain functions of the basic_backend library in the template below
_h denotes this as something of a header file as would be in C
]]
local templet=require("templet")
local init=require("init")
local ffi=require("ffi")
local TYPE=init.TYPE
local M=ffi.load(love.filesystem.getSourceBaseDirectory().."/Prometheus.love/Basic/basic_backend")

ffi.cdef(
"typedef "..TYPE.." TYPE;"..[[
void* _malloc(size_t size);
void* _calloc(size_t num, size_t size);
void* _realloc(void * ptr, size_t new_size);
void _free(void * ptr);
TYPE Vector_Sum(TYPE u[], size_t len);
TYPE Vector_Vector(TYPE u[], TYPE v[], size_t u_len);
void Vector_Vector_Add(TYPE u[], TYPE v[], bool u_trans, bool v_trans, size_t u_len, void* _ret);
void Vector_Vector_Elwise(TYPE u[], TYPE v[], bool u_trans, bool v_trans, size_t u_len, void* _ret);
void Square_Matrix_Vector(TYPE u[], void * _m2, bool m2_trans, size_t size, TYPE ret[]);
void Vector_Square_Matrix_Elwise(TYPE u[], void * _m2, bool u_trans, bool m2_trans, size_t size, void* _ret);
void Vector_Square_Matrix_Add(TYPE u[], void * _m2, bool u_trans, bool m2_trans, size_t size, void* _ret);
void Square_Matrix_Matrix(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret);
void Square_Matrix_Matrix_Elwise(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret);
void Square_Matrix_Matrix_Add(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret);
void Square_Matrix_Matrix_Subtract(void * _m1, void * _m2, bool m1_trans, bool m2_trans, size_t size, void * _ret);
typedef TYPE (*func_ptr)(TYPE);
void Vector_Function(func_ptr F[], TYPE v[], size_t size, size_t start, size_t end, TYPE ret[]);
void Vector_Function_Ovw(func_ptr F[], TYPE v[], size_t start, size_t end);
]])

return M
