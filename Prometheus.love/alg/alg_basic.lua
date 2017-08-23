--[[
This script provides the linear algebra using a basic library coded directly in C with no hardware specific optimizations to compile on a per machine basis like BLAS
There is not so much error checking as these are internal functions and, as effectively wrappers to C functions meant to increase performance, decrease performance enough as it is
]]

local ffi=require "ffi"
local init=require "init"
local templet=require "templet"
local TYPE=init.TYPE

--return loadstring(templet.loadstring[[]])

local alg={backend=backend,__index={}}
alg.backend=require("Basic.basic_backend_h") -- note that the namespace is cached and not the individual library (LuaJIT FFI mostly eliminates overhead for calls to library functions but calls to cached functions naturally have overhead)
backend=alg.backend

function alg.__index.Is_Vector(o)
  --call signature: o:Is_Vector() or alg.Is_Vector(o)
  return o.length~=nil and o.trans~=nil and getmetatable(o)==alg
end

function alg.__index.Is_Matrix(o)
  --call signature: o:Is_Matrix() or alg.Is_Matrix(o)
  return o.nrows~=nil and getmetatable(o)==alg
end

function alg.__index.Is_Function(o)
  return o.trans==nil and getmetatable(o)==alg
end

function alg.__index.dA(self)
  local len=self.nrows
  assert(self:Is_Matrix() and len==self.ncols, "Dualpart Matrix Error: object must be square matrix.")
  local ret=ffi.new('TYPE['..len..']['..len..']')
  backend.Square_dA(self[1],len,ret)
  return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=self.trans}, alg)
end

function alg.__call(self, t)
  --call signature vector v: v{i} returns ith element (indexed from 0)
  --call signature matrix m: m{i,j} returns the element in the ith row and jth column  (indexed from 0)
  --call signature function F: F(v) returns F evaluated on v
  local try1,_=pcall(function (x) return math.fmod(t[1],x) end, 1)
  if self:Is_Vector() then
    assert(try1 and t[1]>=0 and t[1]<self.length, "Vector Index error: Must provide nonnegative integer index within length(indexing for matrices, vectors and functions begins at 0).")
    return self[1][t[1]]
  elseif self:Is_Matrix() then
    local try2,_=pcall(function (x) return math.fmod(t[2],x) end, 1)
    assert(try1 and try2 and t[2]>=0 and t[1]>=0 and t[2]<self.ncols and t[1]<self.nrows, "Matrix Index error: Must provide nonnegative integer indices within range (indexing for matrices, vectors and functions begins at 0).")
    return self[1][t[1]][t[2]]
  elseif self:Is_Function() then
    return alg.Vector_Function{self,t}
  else
    error("Index Error: index out of bounds (indexing for matrices, vectors and functions begins at 0).")
  end
end

function alg.__index.Values(self)--returns the values of a matrix or vector
    --call signature: o:Values() or alg.Values(o)
  return self[1]
end

function alg.Resolve_Pointer(pointer)--just sugar
  return pointer[0]
end

function alg.Vector_Iterator(v)
  local i = 0
  local n = v.length
  return function ()
    if i <= n then
      i = i + 1
      return v[1][i]
    else
      return nil
    end
  end
end

function alg.Matrix_Iterator(m)
  local i = 0
  local j = 0
  local r,c = m.nrows, m.ncols;
  local m=m[1]--locally cache m's values
  return function ()
    if i < r and j<c then
      if j<c then j = j + 1 end
      if j==c-1 then i= i + 1 end
      return m[i][j]
    else
      return nil
    end
  end
end

function alg.Vector_To_String(v)--CURRENTLY DOES NOT FACTOR IN TRANSPOSE
  if v.trans then
    local ret="["
    local len=v.length
    local v=v[1]--locally cache v's values
    for i=0,len-2 do
      ret=ret..v[i]..", "
    end
    return ret..v[len-1].."]"
  else
    local ret="--\n"
    local len=v.length
    local v=v[1]--locally cache v's values
    for i=0,len-2 do
      ret=ret..v[i].."\n"
    end
    return ret..v[len-1].."\n-- \n"
  end
end

function alg.Vector_To_Table(v)--DOESN'T CARE ABOUT TRANSPOSE
  local ret={}
  local len=v.length
  local v=v[1]--locally cache v's values
  for i=1,len do
    ret[i]=v[i-1]
  end
  return ret
end
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--UTILITY FUNCTIONS FOR MATRIX PRINTING... VERY BRUTE FORCE AND LAZY... BUT THEY'RE JUST FOR PRINTING FOR DEBUGGING
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
local function Spaces(n)
  local string=""
  for i=1,n do
    string=string.." "
  end
  return string
end
local function Num_Length(n)
  local ret=#tostring(n)
  if math.fmod(n,1)==0 then
    ret=ret+.5
  end
  if n<0 then
    ret=ret-.5
  end
  return ret
end
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function alg.Matrix_To_String(mat)--DOESN'T CARE ABOUT TRANSPOSE
  local strings={}
  local nrows=mat.nrows
  local ncolumns=mat.ncols
  local max=0
  for j=0,ncolumns-1 do
    max=0
    for i=0,nrows-1 do
      strings[i+1]=strings[i+1] or ''
      temp=#tostring(mat[1][i][j])
      if max<temp then
        max=temp
      end
    end
    for i=0,nrows-1 do
      strings[i+1]=strings[i+1]..mat[1][i][j]..Spaces(2*(max-Num_Length(mat[1][i][j])+2))
    end
  end
  local ret=''
  for i=1,nrows do
    ret=ret..'\n '..strings[i]
  end
  return ret..'\n'
end

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
function alg.Matrix_To_Table(m)--CURRENTLY DOES NOT FACTOR IN TRANSPOSE
  local r,c= m.nrows, m.ncols;
  local ret={}
  local m=m[1]
  for i=0, r-1  do
    ret[i+1]={}
    for j=0, c-1  do
      ret[i+1][j+1]=m[i][j]
    end
  end
  return ret
end

function alg.__tostring(o)
  if o:Is_Vector() then
    return alg.Vector_To_String(o)
  elseif o:Is_Matrix() then
    return alg.Matrix_To_String(o)
  else
    return tostring(o)
  end
end

function alg.New_Vector(t)
  --call signature: alg.New_Vector{length(=) [,trans(=), ptr(=)]} or alg.New_Vector{{1,2,3,...} [,length(=),trans(=)]}
  --NOTE: to get the actual array of values for a vector one must "dereference" it in a sense as one does a regular pointer in the FFI
  --(i.e. ptr[0] retrieves the value(s) pointed to by ptr) EXCEPT it is v[1] not 0
  --in the case where a table Initializer is given with a length,
  if type(t[1])=='table' then
    local trans=t.trans or t[3] or false
    local length=t.length or t[2] or #t[1]
    assert(length>=#t[1], "Vector Initialization Error: Length with given table initializer must be greater than table length.")
    return setmetatable({ffi.new("TYPE ["..length.."]",t[1]), length=length, trans=trans}, alg)
  else
    local trans=t.trans or t[2] or false
    local length=t.length or t[1]
    local ptr=t.ptr or t[3]
    -- can comment out parameter checking for minor performance increase
    local _,c=pcall(function (x) return math.fmod(x,1) end, length)
    assert(c==0, "Vector Initialization Error: Must provide integer length if no table initializer.")
    if not ptr then
      ptr=ffi.new("TYPE [?]",length)
    end
    return setmetatable({ptr, length=length, trans=trans}, alg)
  end
end

function alg.New_Matrix(t)
  --call signature: alg.New_Matrix{nrows(=), ncols(=) [,trans(=), ptr(=)]} or alg.New_Matrix{{{1,2,...},{3,4,...}}, [,nrows(=), ncols(=), trans(=)]}
  --NOTE: to get the actual array of values for a matrix one must "dereference" it in a sense as one does a regular pointer in the FFI
  --(i.e. ptr[0] retrieves the value(s) pointed to by ptr) EXCEPT it is v[1] not 0
  if type(t[1])~='table' then
    local nrows=t.nrows or t[1]
    local ncols=t.ncols or t[2]
    local trans=t.trans or t[3] or false
    local ptr=t.ptr or t[4]
    -- can comment out parameter checking for minor performance increase
    local _,cr=pcall(function (x) return math.fmod(x,1) end, nrows)
    local _,cc=pcall(function (x) return math.fmod(x,1) end, ncols)
    assert(cc==0 and cr==0, "Matrix Initialization Error: Must provide number of rows and number of columns when no table initializer.")
    if not ptr then
      ptr=ffi.new("TYPE ["..nrows.."]".."["..ncols.."]")
    end
      return setmetatable({ptr, nrows=nrows, ncols=ncols, trans=trans}, alg)
  else
    local nrows=t.nrows or t[2] or #t[1]
    local ncols=t.ncols or t[3] or #t[1][1]
    assert(nrows>=#t[1] and ncols>=#t[1][1], "Matrix Initialization Error: When giving table initializer, nrows and ncols must be greater than the corresponding dimensions of the table.")
    local trans=t.trans or t[4] or false
    return setmetatable({ffi.new("TYPE ["..nrows.."]".."["..ncols.."]",t[1]), nrows=nrows, ncols=ncols, trans=trans}, alg)
  end
end

function alg.New_Function(t)
  --call signature: alg.New_Function{length(=), [,ptr]} or alg.New_Function{{fun1, fun2, fun3,...} [,length(=)]}
  --must give a funtion as per alg.functions library
  if type(t[1])=='table' then
    local nfuns=t.length or t[2] or #t[1]
    local ptr=ffi.new('function['..nfuns..']', t[1])
    return setmetatable({ptr, length=nfuns}, alg)
  else
    local nfuns=t.length or t[1]
    local ptr=ffi.new('function[?]',nfuns)
    return setmetatable({ptr, length=nfuns}, alg)
  end
end

function alg.__index.Shape(self)
  if self:Is_Matrix() then
    return self:Matrix_Shape()
  elseif self:Is_Vector() or self:Is_Function() then
    return self.length
  else
    error("Shape Error: Object shape undefined.")
  end
end

function alg.__index.Matrix_Size(m)--number of elements for matrices--nothing more than sugar
  return m.nrows*m.ncols
end

function alg.__index.Matrix_Shape(m)--nothing more than sugar
  return {m.nrows, m.ncols}
end

function alg.__index.Vector_Length(v)--nothing more than sugar
  return v.length
end

function alg.Vector_Vector_Add(t)
  --elementwise add two vectors
  --call signature: alg.Vector_Vector_Add{v1(=), v2(=) [,v1_trans(=), v2_trans(=)]}
  --v1_trans, v2_trans are to override actual states of v1,v2
  --calls C function in basic_backend library
  --Can get slight performance boost doing this manually each  time (i.e. get rid of caching overhead by not using this utility)(?)
  --NOTE: There is an implicit cast to 'void *' in argument 6 of Vector_Vector_Add in each condition
  local u=t.v1 or t[1]
  local v=t.v2 or t[2]
  local u_transposed=t.v1_trans or t[3] or u.trans
  local v_transposed=t.v2_trans or t[4] or v.trans
  local len=u.length
  assert(len==v.length, "Vector Length Error: Vectors must have same lengths to add.")
  local u_transposed= u_transposed or false
  local v_transposed= v_transposed or false
  if(u_transposed) then
    if(v_transposed) then
      local ret=backend._malloc(ffi.sizeof(TYPE)*len)
      backend.Vector_Vector_Add(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE *',ret), backend._free), length=len, trans=true}, alg)
    else
      local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
      backend.Vector_Vector_Add(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
    end
  else
    if(v_transposed) then
      local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
      backend.Vector_Vector_Add(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
    else
      local ret=backend._malloc(ffi.sizeof(TYPE)*len)
      backend.Vector_Vector_Add(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE *',ret), backend._free), length=len, trans=false} ,alg)
    end
  end
end

function alg.Vector_Vector(t)
  --dot product
  --call signature: alg.Vector_Vector{v1(=), v2(=)}
  --calls C function in basic_backend library
  --Can get slight performance boost doing this manually each  time (i.e. get rid of caching overhead by not using this utility)(?)
  local u=t.v1 or t[1]
  local v=t.v2 or t[2]
  local len=u.length
  assert(len==v.length, "Vector Length Error: Vectors must have same lengths to add.")
  return backend.Vector_Vector(u,v,size)
end

function alg.Vector_Vector_Elwise(t)--elementwise multiply two vectors. This utility function checks compatibility and handles garbage collection.
  --elementwise multiply two vectors
  --call signature: alg.Vector_Vector_Add{v1(=), v2(=) [,v1_trans(=), v2_trans(=)]}
  --v1_trans, v2_trans are to override actual states of v1,v2
  --calls C function in basic_backend library
  --Can get slight performance boost doing this manually each  time (i.e. get rid of caching overhead by not using this utility)(?)
  --NOTE: There is an implicit cast to 'void *' in argument 6 of Vector_Vector_Add in each condition
  local u=t.v1 or t[1]
  local v=t.v2 or t[2]
  local u_transposed=t.v1_trans or t[3] or u.trans
  local v_transposed=t.v2_trans or t[4] or v.trans
  local len=u.length
  assert(len==v.length, "Vector Length Error: Vectors must have same lengths to multiply elementwise.")
  local u_transposed= u_transposed or false
  local v_transposed= v_transposed or false
  if(u_transposed) then
    if(v_transposed) then
      local ret=backend._malloc(ffi.sizeof(TYPE)*len)
      backend.Vector_Vector_Elwise(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE *',ret), backend._free), length=len, trans=true}, alg)
    else
      local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
      backend.Vector_Vector_Elwise(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']',ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
    end
  else
    if(v_transposed) then
      local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
      backend.Vector_Vector_Elwise(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']',ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
    else
      local ret=backend._malloc(ffi.sizeof(TYPE)*len)
      backend.Vector_Vector_Elwise(u[1], v[1], u_transposed, v_transposed, len, ret)
      return setmetatable({ffi.gc(ffi.cast('TYPE *',ret), backend._free), length=len, trans=false} ,alg)
    end
  end
end

function alg.Square_Matrix_Vector(t)
  --NOTE: Does not consider whether the vector is transposed or not
  --call signature: alg.Square_Matrix_Vector{m(=), v(=) [,m_trans(=)]}
  local m=t.m or t[1]
  local v=t.v or t[2]
  local len=v.length
  local trans=t.m_trans or t[3] or m.trans
  assert(len==m.nrows and len==m.ncols, "Square Matrix - Vector Multiplication Error: Matrix must be square and have number of columns equal to the length of the vector.")
  local ret=backend._calloc(len,ffi.sizeof(TYPE))
  backend.Square_Matrix_Vector(v[1],m[1],trans,len,ret)
  return setmetatable({ffi.gc(ffi.cast('TYPE *',ret), backend._free), length=len, trans=false}, alg)
end

function alg.Vector_Square_Matrix_Elwise(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{v(=), m(=) [,v_trans(=), m_trans(=), overwrite(=)]}
  local m=t.m or t[2]
  local v=t.v or t[1]
  local v_transposed=t.v_trans or t[3] or v.trans
  local len=v.length
  local m_transposed=t.m_trans or t[4] or m.trans
  assert(len==m.nrows and len==m.ncols, "Vector-Square Matrix Elementwise Multiplication Error: Matrix must be square and have number of columns equal to the length of the vector.")
  if t.overwrite or t[5] then
    backend.Vector_Square_Matrix_Elwise(v[1], m[1], v_transposed, m_transposed, len, m[1])
    return m
  else
    local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
    backend.Vector_Square_Matrix_Elwise(v[1], m[1], v_transposed, m_transposed, len, ret)
    return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']',ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
  end
end

function alg.Vector_Square_Matrix_Add(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{v(=), m(=) [,v_trans(=), m_trans(=), overwrite(=)]}
  local m=t.m or t[2]
  local v=t.v or t[1]
  local v_transposed=t.v_trans or t[3] or v.trans
  local len=v.length
  local m_transposed=t.m_trans or t[4] or m.trans
  assert(len==m.nrows and len==m.ncols, "Vector-Square Matrix Elementwise Addition Error: Matrix must be square and have number of columns equal to the length of the vector.")
  if t.overwrite or t[5] then
    backend.Vector_Square_Matrix_Add(v[1], m[1], v_transposed, m_transposed, len, m[1])
    return m
  else
    local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
    backend.Vector_Square_Matrix_Add(v[1], m[1], v_transposed, m_transposed, len, ret)
    return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']',ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
  end
end

function alg.Square_Matrix_Matrix(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{m1(=), m2(=) [,m1_trans(=), m2_trans(=)]}
  local m1=t.m1 or t[1]
  local m2=t.m2 or t[2]
  local len=m1.ncols
  local m1_transposed=t.m1_trans or t[3] or m1.trans
  local m2_transposed=t.m2_trans or t[4] or m2.trans
  assert(len==m1.nrows and len==m2.ncols and m2.nrows==len, "Square Matrix-Matrix Multiplication Error: Matrices must be square and have same shape.")
  local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
  backend.Square_Matrix_Matrix(m1[1], m2[1], m1_transposed, m2_transposed, len, ret)
  return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']',ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
end

function alg.Square_Matrix_Matrix_Elwise(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{m1(=), m2(=) [,m1_trans(=), m2_trans(=), overwrite(=)]}
  local m1=t.m1 or t[1]
  local m2=t.m2 or t[2]
  local len=m1.ncols
  local m1_transposed=t.m1_trans or t[3] or m1.trans
  local m2_transposed=t.m2_trans or t[4] or m2.trans
  assert(len==m1.nrows and len==m2.ncols and m2.nrows==len, "Square Matrix-Matrix Elementwise Multiplication Error: Matrices must be square and have same shape.")
  if t.overwrite or t[5] then
    --overwrite m2
    backend.Square_Matrix_Matrix_Elwise(m1[1], m2[1], v_transposed, m_transposed, len, m2[1])
    return m2
  else
    local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
    backend.Square_Matrix_Matrix_Elwise(m1[1], m2[1], m1_transposed, m2_transposed, len, ret)
    return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
  end
end

function alg.Square_Matrix_Matrix_Add(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{m1(=), m2(=) [,m1_trans(=), m2_trans(=)]}
  local m1=t.m1 or t[1]
  local m2=t.m2 or t[2]
  local len=m1.ncols
  local m1_transposed=t.m1_trans or t[3] or m1.trans
  local m2_transposed=t.m2_trans or t[4] or m2.trans
  assert(len==m1.nrows and len==m2.ncols and m2.nrows==len, "Square Matrix-Matrix Elementwise Addition Error: Matrices must be square and have same shape.")
  local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
  backend.Square_Matrix_Matrix_Add(m1[1], m2[1], m1_transposed, m2_transposed, len, ret)
  return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
end

function alg.Square_Matrix_Matrix_Subtract(t)
  --Is commutative (vector and matrix must still be in the proper arguments)
  --call signature: alg.Vector_Square_Matrix_Elwise{m1(=), m2(=) [,m1_trans(=), m2_trans(=)]}
  local m1=t.m1 or t[1]
  local m2=t.m2 or t[2]
  local len=m1.ncols
  local m1_transposed=t.m1_trans or t[3] or m1.trans
  local m2_transposed=t.m2_trans or t[4] or m2.trans
  assert(len==m1.nrows and len==m2.ncols and m2.nrows==len, "Square Matrix-Matrix Elementwise Subtraction Error: Matrices must be square and have same shape.")
  local ret=backend._malloc(ffi.sizeof(TYPE)*len*len)
  backend.Square_Matrix_Matrix_Subtract(m1[1], m2[1], m1_transposed, m2_transposed, len, ret)
  return setmetatable({ffi.gc(ffi.cast('TYPE (*)['..len..']', ret), backend._free), nrows=len, ncols=len, trans=false}, alg)
end

function alg.Vector_Function(t)
  --BIG NOTE: IF YOU USE A FUNCTION WITH LUA FUNCTIONS THEN YOU ARE TAKING ON SUBSTANTIAL OVERHEAD OF C TO LUA CALLBACKS
  --call signature: alg.Vector_Function{F,v [, f_start(=), f_end(=), overwrite(=), derivative(=)]}
  --F is array of functions, v is vector to apply functions to Elementwise, f_start is beginning of contiguous region in array to apply functions to, f_end ..., overwrite writes over the memory of the input vector, derivative tells whether you want the to apply the first derivatives or not.
  local F=t.F or t[1]
  local v=t.v or t[2]
  local len=F.length
  assert(len==v.length, "Vector-Function Error: Length of vector of functions must match that of value vector.")
  local f_start=t.f_start or t[3] or 0
  local f_end=t.f_end or t[4] or  len-1
  assert(not f_start~=0 or f_end, "Function Application Bounds Error: Given start bound must too provide end bound.")
  local overwrite=t.overwrite or t[5]
  local derivative=t.derivative or t[6] or false
  if overwrite then
    backend.Vector_Function(F[1], v[1], len, f_start, f_end, v[1], derivative)
    return v
  else
    local ret=backend._malloc(ffi.sizeof(TYPE)*len)
    backend._memcpy(ret,v[1],ffi.sizeof(TYPE)*len)
    backend.Vector_Function(F[1], v[1], len, f_start, f_end, ret, derivative)
    return setmetatable({ffi.gc(ffi.cast('TYPE *', ret), backend._free), length=len, trans=false}, alg)
  end
end

local alg_mt={}

function alg_mt.__call(self,t)
  --adds the ability to call the library itself to generate vectors, matrices, or functions
  --call signature: called the same way as New_Vector, etc EXCEPT you must include a string hashed as 'new' telling of what structure you want of 'vector','matrix','function'(and the constructor follows the same syntax as the respective New_)
  local kind=t.new
  assert(type(kind)=='string', "Alg_Basic New Call Error: Must give value as string for new= (case insensitive) from 'vector','matrix','function'.")
  kind=kind:lower()
  assert(kind=='vector' or kind=='matrix' or kind=='function', "Alg_Basic New Call Error: new=  must be (case insensitive) from 'vector','matrix','function'.")
  if kind=='vector' then
    return alg.New_Vector(t)
  elseif kind=='matrix' then
    return alg.New_Matrix(t)
  else
    return alg.New_Function(t)
  end
end

setmetatable(alg, alg_mt)

return alg
