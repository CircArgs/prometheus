--[[
class for networks
TODO: currently can only call a network to evaluate on a vector. need to add matrices (larger batched than 1 i.e. a single vector)
TODO TODO TODO: implement the entire network Evaluation procedure in C to eliminate overhead of calls in Lua.
THIS WILL BE MOST IMPORTANT FOR GPGPU FOR YOU DONT WANT TO KEEP LOADING THE DATA ONTO THE GPU EACH STEP

TODO TODO: USING GENERAL LINEAR ALGEBRA OPERATIONS IS VERY INNEFICIENT ESPECIALLY IN THE CASE WHERE WE DIFFERENTIATE WITH RESPECT TO THE LINK WEIGHTS. WANT SPARSE MATRIX multiplication AND ADDITION WITH BOTH IGNORING 0 ELEMENTS IN dA
]]

local init=require "init"
local alg=require "alg.alg_basic"
local ffi=require"ffi"
local TYPE=init.TYPE
local network={__index={}}
local network_mt={}

--[[
local pair={}

function pair.__call(self, o)


]]

function network.__call(self, v, grad)
  assert(v:Is_Vector(), "Network Evaluation Error: Networks can currently only be evaluated on vectors.")
  local F=self[2]
  local A=self[1]--get matrix (a.k.a. 'structure')
  local end_range=self.ninputs+self.nhidden-1
  local ninputs=self.ninputs
  if not grad then
    for i=1, self.nhidden do
      v=alg.Square_Matrix_Vector{A, v}--during a matrix-vector multiplication we cannot overwrite the original (but setting the original pointer to this new one will have the original garbage collected)
      alg.Vector_Function{F, v, ninputs, end_range, overwrite=true}
    end
    v=alg.Square_Matrix_Vector{A, v}
    alg.Vector_Function{F, v, overwrite=true}
    return v
  else
    local dA=A:dA()
    local At1=alg.backend._malloc(ffi.sizeof(TYPE)*A.nrows) --A^t[1] i.e. sum along columns
    alg.backend.Sum_Dir(A[1], A.nrows, A.nrows, 1, At1)
    setmetatable({ffi.gc(ffi.cast('TYPE *', At1), backend._free), length=A.nrows, trans=false}, alg)
    --notes use ** as Hadamard product
    --this can definitely be prettified using the abstraction of an ordered pair from the pdf... just didnt do it (would have a bit more overhead with table hashing and __call call)
    alg.Vector_Square_Matrix_Elwise{v, dA, true, overwrite=true}--v^t ** dA
    alg.Square_Matrix_Vector{A, v}--Av
    alg.Vector_Square_Matrix_Elwise{alg.Vector_Function{F, v, ninputs, end_range, derivative=true}, dA, overwrite=true}--F'(Av) ** (v^t ** dA)
    alg.Vector_Function{F, v, ninputs, end_range, overwrite=true} --F(Av)
    for i=2, self.nhidden do--note starting at 2 because one step done already
      alg.Vector_Square_Matrix_Add{v, alg.Vector_Square_Matrix_Elwise{At1, dA, true, overwrite=true}, true, overwrite=true}--(F(Av)^t + A^t[1] ** F'(Av) ** v^t) ** dA
      alg.Square_Matrix_Vector{A, v} --AF(Av)
      alg.Vector_Square_Matrix_Elwise{alg.Vector_Function{F, v, ninputs, end_range, derivative=true}, dA, overwrite=true}--(F'(AF(Av))**(F(Av)^t + A^t[1] ** F'(Av) ** v^t)) ** dA
      alg.Vector_Function{F, v, ninputs, end_range, overwrite=true}--F(AF(AV))
      --repeat...
    end
    alg.Vector_Square_Matrix_Elwise{alg.Vector_Function{F, v, ninputs, end_range, derivative=true}, dA, overwrite=true}--G'(AF(AF(AF....))) ** ... ** dA
    alg.Square_Matrix_Matrix_Elwise{A:dA(), dA, overwrite=true}
    alg.Vector_Function{F, v, overwrite=true}
    --we overwrote the input vector and dA when we did stuff so we just return pointers to them (dA wasnt a matrix though so we make it one before returning)
    return v, {dA, nrows=A.nrows, ncols=A.ncols, trans=false}
  end
end

function network.__index.backprop(self,m)
  --m is a square matrix with the same shape as the structure of self
  --call signature: net:backprop(m)
  alg.Square_Matrix_Matrix_Subtract{self[1],m}
end

function network.New_Network(t)
  --call signature: network.New_Network{structure(=), activations(=), ninputs(=), noutputs(=), nhidden(=)}
  local structure=t.structure or t[1]
  assert(structure:Is_Matrix() and structure.nrows==structure.ncols, "Newtwork Initialization Error: Structure must be square matrix.")
  local size=structure.nrows
  local activations=t.activations or t[2]
  assert(activations:Is_Function() and structure.nrows==structure.ncols, "Newtwork Initialization Error: Activations must be square matrix.")
  local ninputs=t.ninputs or t[3]
  local nhidden=t.nhidden or t[5]
  local noutputs=t.noutputs or t[4]
  local _,c1=pcall(function(x) return math.fmod(x,1) end, ninputs)
  local _,c2=pcall(function(x) return math.fmod(x,1) end, nhidden)
  local _,c3=pcall(function(x) return math.fmod(x,1) end, noutputs)
  assert(c1==0 and c2==0 and c3==0, "Newtwork Initialization Error: ninputs and noutputs must be positive INTEGERS and nhidden must be a nonnegative INTEGER.")
  assert(ninputs>0 and nhidden>-1 and noutputs>0, "Newtwork Initialization Error: ninputs and noutputs must be POSITIVE integers and nhidden must be a NONNEGATIVE integer.")
  assert(ninputs+nhidden+noutputs==size, "Newtwork Initialization Error: ninputs+nhidden+noutputs must equal the nrows and ncols of the given structure.")
  return setmetatable({structure, activations, structure=structure, activations=activations, size=size, ninputs=ninputs, nhidden=nhidden, noutputs=noutputs},network)
end

function network.__index.Is_Network(self)
  return network==getmetatable(self)
end

function network_mt.__call(self, t)
  --can call library to make new network with otherwise identical table syntax
  --see New_Network for call signature
  return network.New_Network(t)
end

setmetatable(network, network_mt)

return network
