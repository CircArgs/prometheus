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
local functions=require"functions"
local TYPE=init.TYPE
local network={__index={}}
local network_mt={}

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
    alg.Vector_Square_Matrix_Elwise{v, dA, v_trans=true, overwrite=true}--v^t ** dA
    v=alg.Square_Matrix_Vector{A, v}--Av
    alg.Vector_Square_Matrix_Elwise{alg.Vector_Function{F, v, ninputs, end_range, derivative=true}, dA, overwrite=true}--F'(Av) ** (v^t ** dA)--_/
    alg.Vector_Function{F, v, ninputs, end_range, overwrite=true} --F(Av)
    for i=2, self.nhidden do--note starting at 2 because one step done already
      alg.Vector_Square_Matrix_Add{v, alg.Vector_Square_Matrix_Elwise{At1, dA, true, overwrite=true}, true, overwrite=true}--(F(Av)^t + A^t[1] ** F'(Av) ** v^t) ** dA
      v=alg.Square_Matrix_Vector{A, v} --AF(Av)
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

function network.__index.backprop(self, m)
  --m is a square matrix with the same shape as the structure of self
  --m is meant to be a gradient shaped into a matrix of the derivatives with respect to the weights of the correspondingadjaceny matrix of the network
  --call signature: net:backprop(m)
  alg.Square_Matrix_Matrix_Subtract{self[1], m}
end

function network.New_Network(t)
  --call signature: network.New_Network{structure(=), activations(=), ninputs(=), noutputs(=), nhidden(=)}
  --structure is an adjacency matrix
  --activations is a vector of functions to serve as their respective row's (each a node's array of link weights) activation function
  --size is the total number of nodes in the network
  --ninputs, nhidden, noutputs are self explanatory
  --memories is a vector of outputs from output nodes not in the original problem structure (i.e. developed via mutation to feed into the network over time condensing historic behavior to potentially influence future behavior)
  local structure=t.structure or t[1]
  assert(structure:Is_Matrix() and structure.nrows==structure.ncols, "Newtwork Initialization Error: Structure must be square matrix.")
  local size=structure.nrows
  local activations=t.activations or t[2]
  assert(activations:Is_Function() and structure.nrows==structure.ncols, "Newtwork Initialization Error: Activations must be square matrix.")
  local ninputs=t.ninputs or t[3]
  local nhidden=t.nhidden or t[5]
  local noutputs=t.noutputs or t[4]
  local _,c1=pcall(function(x) return math.fmod(x, 1) end, ninputs)
  local _,c2=pcall(function(x) return math.fmod(x, 1) end, nhidden)
  local _,c3=pcall(function(x) return math.fmod(x, 1) end, noutputs)
  assert(c1==0 and c2==0 and c3==0, "Newtwork Initialization Error: ninputs and noutputs must be positive INTEGERS and nhidden must be a nonnegative INTEGER.")
  assert(ninputs>0 and nhidden>-1 and noutputs>0, "Newtwork Initialization Error: ninputs and noutputs must be POSITIVE integers and nhidden must be a NONNEGATIVE integer.")
  assert(ninputs+nhidden+noutputs==size, "Newtwork Initialization Error: ninputs+nhidden+noutputs must equal the nrows and ncols of the given structure.")
  return setmetatable({structure, activations, structure=structure, activations=activations, size=size, ninputs=ninputs, nhidden=nhidden, noutputs=noutputs, memories=nil} network)
end

function network_mt.__call(self, t)
  --can call library to make new network with otherwise identical table syntax
  --see New_Network for call signature
  return network.New_Network(t)
end

function network.__index.Is_Network(self)
  return network==getmetatable(self)
end

function network.__index.Add_Node(self, t)
  --call signature: A:Add_Node{ins(=), outs(=), fun(=) [, in_weights(=), out_weights(=)]}
  --ins is a table of input indices, outs is a table of output indices, function is the activation function of the node
  --if ins=nil then it is assumed the node is a new input in which case a corresponding output node will be added to the network
  --in_weights is a table of weights which will (in order) be assigned to the new links from ins to the new node
  --out_weights is a table of weights which will (in order) be assigned to the new links from outs to the new node
  --#outs==#ins or given in_weights and out_weights where #ins==#in_weights or #out_weights==#outs
  --in other words, we assume that if #ins=#outs, those links are being replaced (ins[i] and outs[i] currently have a link that will be set to 0 once the new node is added). Otherwise, we do not
  --if ins is nil and out_weights is nil then the links are initilized to 1s
  local ins=t[1] or t.ins or nil
  local tcheck=function(io, ints)--ints (do they have to be integers?) 1 for true 0 for false
    local ints=ints or 1
    for i=1,#io do
      if(not(type(io[i])=='number' and ints*math.fmod(io[i],1)==0 and io[i]>=0)) then
        return false
      end
    end
    return false
  end
  ins==nil or assert(type(ins)=='table' and tcheck(ins), "Add Node Error: Must provide at least one in node and in node(s) must be all nonnegative integers less than the size of the network.")
  local outs=t[2] or t.outs
  assert(type(outs)=='table' and tcheck(outs), "Add Node Error: Must provide at least one out node and out node(s) must be all nonnegative integers less than the size of the network.")
  local in_weights=t[4] or t.in_weights
  local out_weights=t[4] or t.out_weights
  assert(type(in_weights)=='table' and tcheck(in_weights), "Add Node Error: Provided in_weights must be table.")
  assert(type(out_weights)=='table' and tcheck(out_weights), "Add Node Error: Provided in_weights must be table.")
  assert(#outs==#ins or in_weights~=nil and out_weights~=nil and #in_weights=#ins and #out_weights=#outs, "Add Node Error: If same number of in nodes and out nodes not given, must provide in_weights and out_weights with same number of elements respectively.")
  local fun=t[3] or t.fun
  assert(ffi.istype('function', fun), "Add Node Error: fun must be function.")
  if(ins~=nil) then
    --new hidden node
    local new_node_num=self.ninputs+self.nhidden
    local len=self.size
    local mat=self[1][1]
    local F=self[2][1]
    mat=ffi.gc(ffi.cast('TYPE (*)['..len..']', alg.backend._realloc(mat, len*ffi.sizeof(TYPE)*(len+1))))
    F=ffi.gc(ffi.cast('function['..len..']', alg.backend._realloc(F, ffi.sizeof('function')*(len+1))))
    local tempfun=F[new_node_num]
    F[new_node_num]=fun
    for i=new_node_num+1, self.size+1 do
      fun=F[i]
      F[i]=tempfun
      tempfun=fun
    end
    if #ins=#outs then
      for i=1, #ins do
        mat[new_node_num][ins[i]]=in_weights[i] or 1
        mat[outs[i]][new_node_num]=out_weights[i] or mat[outs[i]][ins[i]] or 0
        mat[outs[i]][ins[i]]=0
      end
    else
      for i=1, #ins do
        mat[new_node_num][ins[i]]=in_weights[i] or 1
      end
      for i=1, #outs do
        mat[outs[i]][new_node_num]=out_weights[i] or 0
      end
    end
  else
    --new inputs/outputs
    
    self.nhidden=self.nhidden+1
    self.size=self.size+1
end

function network.Distance(A, B)
  --call signature: network.Distance(A, B)
  --calculates the distance between two networks A and B
  --distance method set in init.runtime.network_distance
  --choices of 'monte carlo' (default) TODO: finish 'Frobenius' (C function already written in basic_backend.c just needs avg function difference and fitness difference), monte carlo submethod: kullbach leibler divergence in case of softmax output, ...more options/combinations?
  --TODO: caching of some outputs and/or inputs to refine monte carlo approach
  --NOTE: RELATED TO THE TODO JUST ABOVE, MONTE CARLO APPROACH IS NOT A GOOD WAY TO WORK WHEN NETWORKS HAVE MEMORY. IT CANNOT TAKE THIS INTO ACCOUNT SINCE MEMORIES ARE TASK SPECIFIC
  --Monte Carlo: random sample of inputs from given function (or default standard normal) some given number of iterations (default 1000)
  assert(A:Is_Network() and B:Is_Network(), "Network Distance Error: Both elements must be networks.")
  local method=init.runtime.network_distance
  if method[1]=='monte carlo' then
    local iters=method.iters or 1000
    local fun = method.fun or function() return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) / 2 end, math.min(A.ninputs, B.ninputs)
    local x=nil
    local ret=0;
    local diff=nil;
    for i=1, method.iters do
      x=alg.New_Vector(fun)
      diff=alg.Vector_Vector_Subtract(A(X), B(X))
      ret=ret+math.math.sqrt(alg.Vector_Sum(alg.Vector_Vector_Elwise(diff,diff)))
    end
    return ret
  else
    error('Network Distance Error: Method undefined.')
  end
end

function network.__sub(A, B)
  --call signature: A-B
  return network.Distance(A,B)
end



setmetatable(network, network_mt)

return network
