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
local functions, _=require("alg.functions")()
local TYPE=init.TYPE
local network={__index={}}
local network_mt={}

function network.__call(self, v, grad, rawout)
  --call signature: Net(v [, grad, rawout])
  --v is the input vector (just the length of the number of problem inputs)
  --grad is a boolean for wanting the gradient with respect to network weight or not
  --rawout is a boolean which says whether an output vector containing the entire final state of the network or not is desired
  local prep_input_vector=function(v)
    local ret=ffi.new('TYPE[?]', self.size)
    for i=0, v.length-1 do--load new inputs
      ret[i]=v[1][i]
    end
    for i=0, self.ninputs-self.nmemories-1 do --load memories
      ret[i+v.length]=self.memories[1][i]
    end
    return setmetatable({ret, length=self.size, trans=v.trans}, alg)
  end
  local prep_outout_vector=function(v)
    local ret=ffi.new('TYPE[?]', self.noutputs-self.nmemories)
    for i=self.size-self.nmemories, self.size-1 do --unload memories
      self.memories[1][i-self.size+self.nmemories]=v[1][i]
    end
    for i=0, self.noutputs-self.nmemories-1 do--unload outputs
      ret[i]=v[1][i+self.size-self.noutputs]
    end
    return setmetatable({ret, length=self.noutputs-self.nmemories, trans=v.trans}, alg)
  end
  assert(v:Is_Vector(), "Network Evaluation Error: Networks can currently only be evaluated on vectors.")
  v=prep_input_vector(v)--transform the inputs into a vector of length size of the network's structure
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
    if(not rawout) then v=prep_outout_vector(v) end
    return v
  else
    local dA=self:dA()
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
    alg.Square_Matrix_Matrix_Elwise{self:dA(), dA, overwrite=true}
    alg.Vector_Function{F, v, overwrite=true}
    --we overwrote the input vector and dA when we did stuff so we just return pointers to them (dA wasnt a matrix though so we make it one before returning)
    if(not rawout) then v=prep_outout_vector(v) end
    return v, setmetatable({dA, nrows=A.nrows, ncols=A.ncols, trans=false}, alg)
  end
end

function network.__index.Erase_Memories(self)
  --call signature: Net:Erase_Memories()
  --resets all the memories of a network to 0s
  self.memories=alg.New_Vector{self.nmemories}
end

function network.__index.dA(self)
  --takes a network and returns its Dualpart matrix
  local mat=self[1]
  local len=mat.nrows
  assert(mat:Is_Matrix() and len==mat.ncols, "Dualpart Fetch Error: Network's structure must be square matrix.")
  local ret=ffi.new('TYPE['..len..']['..len..']')
  for i=0, len-1 do
    for j=0, len-1 do
      if mat[1][i][j]~=0 and (i~=j or (i==j and i<self.ninputs+self.nhidden)) then
        ret[i][j]=1
      end
    end
  end
  return setmetatable({ret, nrows=len, ncols=len, trans=mat.trans}, alg)
end

function network.__index.backprop(self, m)
  --m is a square matrix with the same shape as the structure of self
  --m is meant to be a gradient shaped into a matrix of the derivatives with respect to the weights of the corresponding adjaceny matrix of the network
  --call signature: net:backprop(m)
  alg.Square_Matrix_Matrix_Subtract{self[1], m}
end

function network.New_Network(t)
  --call signature: network.New_Network{structure(=), activations(=), ninputs(=), noutputs(=), nhidden(=) [, nmemories(=)]}
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
  local nmemories=t.nmemories or t[6] or 0
  local _,c1=pcall(function(x) return math.fmod(x, 1) end, ninputs)
  local _,c2=pcall(function(x) return math.fmod(x, 1) end, nhidden)
  local _,c3=pcall(function(x) return math.fmod(x, 1) end, noutputs)
  assert(c1==0 and c2==0 and c3==0, "Newtwork Initialization Error: ninputs and noutputs must be positive INTEGERS and nhidden must be a nonnegative INTEGER.")
  assert(ninputs>0 and nhidden>-1 and noutputs>0, "Newtwork Initialization Error: ninputs and noutputs must be POSITIVE integers and nhidden must be a NONNEGATIVE integer.")
  assert(ninputs+nhidden+noutputs+nmemories==size, "Newtwork Initialization Error: ninputs+nhidden+noutputs+nmemories must equal the nrows and ncols of the given structure.")
  local ret= setmetatable({structure, activations, structure=structure, activations=activations, size=size, ninputs=ninputs, nhidden=nhidden, noutputs=noutputs, nmemories=nmemories, fitness=0}, network)
  ret:Erase_Memories()--set memories to be a vector of zeros
  return ret
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
  --call signature: A:Add_Node{ins(=), outs(=), fun(=) [, in_weights(=), out_weights(=), keep_memories(=)]}
  --ins is a table of input indices, outs is a table of output indices, function is the activation function of the node (the output in case ins=nil i.e. memory input/output pair)
  --[[if ins==nil then it is assumed the node is a new input in which case a corresponding output node will be added to the network.
   outs is then a table of intermediate nodes in the network. in_weights will feed into them and out_weights will feed to the outs.
   All is optional. An empty function call will suffice to add a new memory node where a new input is added and a
   corresponding output are added and linked with weight 1 ]]
  --in_weights is a table of weights which will (in order) be assigned to the new links from ins to the new node
  --out_weights is a table of weights which will (in order) be assigned to the new links from outs to the new node
  --#outs==#ins or given in_weights and out_weights where #ins==#in_weights or #out_weights==#outs
  --in other words, we assume that if #ins==#outs, those links are being replaced (ins[i] and outs[i] currently have a link that will be set to 0 once the new node is added). Otherwise, we do not
  --if ins is nil and out_weights is nil then the links are initilized to 1s
  --keep_memories is a boolean which if false resizes memory array in case ins==nil (new memory node). Otherwise, will copy existing memory values. false by default
  local tcheck=function(io, ints)--ints (do they have to be integers?) 1 (or nothing) for true 0 for false
    local ints=ints or 1
    for i=1,#io do
      if(not(type(io[i])=='number' and ints*math.fmod(io[i], 1)==0 and io[i]>=0 and io[i]<self.size)) then
        return false
      end
    end
    return true
  end
  local shift=function(mat, new_node_num, len, opt)
    local opt=opt or 0
    if opt==0 then
      local temp=ffi.gc(ffi.cast('TYPE (*)['..(len+1)..']', alg.backend._malloc(ffi.sizeof(TYPE)*(len+1)*(len+1))), alg.backend._free)
      for i=0, new_node_num-1 do
        temp[i][new_node_num]=0
        temp[new_node_num][i]=0
        for j=0, new_node_num-1 do
          temp[i][j]=mat[i][j]
        end
      end
      for i=new_node_num+1, len do
        temp[i][new_node_num]=0
        temp[new_node_num][i]=0
        for j=0, new_node_num-1 do
          temp[i][j]=mat[i-1][j]
        end
      end
      for i=0, new_node_num-1 do
        for j=new_node_num+1, len do
          temp[i][j]=mat[i][j-1]
        end
      end
      for i=new_node_num+1, len do
        for j=new_node_num+1, len do
          temp[i][j]=mat[i-1][j-1]
        end
      end
      temp[new_node_num][new_node_num]=0
      return temp
    else
      local temp=ffi.gc(ffi.cast('TYPE (*)['..(len+2)..']', alg.backend._malloc(ffi.sizeof(TYPE)*(len+2)*(len+2))), alg.backend._free)
      for i=0, new_node_num-1 do
        temp[i][new_node_num]=0
        temp[new_node_num][i]=0
        temp[len][i]=0
        temp[i][len]=0
        temp[len+1][i]=0
        temp[i][len+1]=0
        for j=0, new_node_num-1 do
          temp[i][j]=mat[i][j]
        end
      end
      for i=new_node_num+1, len do
        temp[i][new_node_num]=0
        temp[new_node_num][i]=0
        temp[len][i]=0
        temp[i][len]=0
        temp[len+1][i]=0
        temp[i][len+1]=0
        for j=0, new_node_num-1 do
          temp[i][j]=mat[i-1][j]
        end
      end
      for i=0, new_node_num-1 do
        for j=new_node_num+1, len do
          temp[i][j]=mat[i][j-1]
        end
      end
      for i=new_node_num+1, len do
        for j=new_node_num+1, len do
          temp[i][j]=mat[i-1][j-1]
        end
      end
      temp[new_node_num][new_node_num]=0
      temp[new_node_num][len]=0
      temp[new_node_num][len+1]=0
      temp[len+1][new_node_num]=0
      temp[len][new_node_num]=0
      temp[len+1][len+1]=1
      return temp
    end
  end
  local fun_cpy_insert=function(F, fun1, ind1, len, fun2, ind2)
    local ret=ffi.new('function[?]', len)
    for i=0, ind1-1 do
      ret[i]=F[i]
    end
    ret[ind1]=fun1
    for i=ind1+1, len-2 do
      ret[i]=F[i-1]
    end
    if ind2 then
      ret[len-1]=fun2
    else
      ret[len-1]=F[len-2]
    end
    return ret
  end
  local ins=t[1] or t.ins or nil
  assert(ins==nil or type(ins)=='table' and #ins>0 and tcheck(ins), "Add Node Error: Must provide at least one in node (or ins=nil) and in node(s) must be all nonnegative integers less than the size of the network.")
  local outs=t.outs or t[2]  or {}
  assert(type(outs)=='table' and #outs>=0 and tcheck(outs), "Add Node Error: Must provide at least one out node and out node(s) must be all nonnegative integers less than the size of the network.")
  local in_weights=t.in_weights or t[4] or  {}
  local out_weights=t.out_weights or t[5] or {}
  assert(type(in_weights)=='table' and tcheck(in_weights, 0), "Add Node Error: Provided in_weights must be table.")
  assert(type(out_weights)=='table' and tcheck(out_weights, 0), "Add Node Error: Provided out_weights must be table.")
  local fun=t.fun or t[3] or functions.linear
  assert(ffi.istype('function', fun), "Add Node Error: fun must be function.")
  local len=self.size
  if(ins~=nil) then
    assert(#outs==#ins or #in_weights>0 and #out_weights>0 and #in_weights==#ins and #out_weights==#outs, "Add Node Error: If same number of in nodes and out nodes not given, must provide in_weights and out_weights with same number of elements respectively.")
    --new hidden node
    local new_node_num=self.ninputs+self.nhidden
    self[1][1]=shift(self[1][1], new_node_num, len)
    local mat=self[1][1]
    self[2][1]=fun_cpy_insert(self[2][1], fun, new_node_num, len+1)
    local F=self[2][1]
    self[2].length=self[2].length+1--activations is longer
    self[1].nrows=self[1].nrows+1--structure larger
    self[1].ncols=self[1].ncols+1
    local shift_check=function(x)--increments node number if has been shifted by shift
      if self.ninputs+self.nhidden<=x then
        return x+1
      end
      return x
    end
    if #ins==#outs then
      for i=1, #ins do
        mat[new_node_num][shift_check(ins[i])]=in_weights[i] or 1
        mat[shift_check(outs[i])][new_node_num]=out_weights[i] or mat[outs[i]][ins[i]] or 0
        mat[shift_check(outs[i])][shift_check(ins[i])]=0
      end
    else
      for i=1, #ins do
        mat[new_node_num][shift_check(ins[i])]=in_weights[i] or 1
      end
      for i=1, #outs do
        mat[shift_check(outs[i])][new_node_num]=out_weights[i] or 0
      end
    end
    self.nhidden=self.nhidden+1
  else
    --can provide number of out_weights as number of outs (intermediates) or as number of intermediates and last will go to corresponding new out for memory
    --new inputs/outputs
    local new_node_num=self.ninputs
    self[1][1]=shift(self[1][1], new_node_num, len, 1)
    local mat=self[1][1]
    self[2][1]=fun_cpy_insert(self[2][1], functions.linear, new_node_num, len+2, fun, "pigglywiggly")
    local F=self[2][1]
    self[2].length=self[2].length+2--activations longer
    self[1].nrows=self[1].nrows+2--structure larger
    self[1].ncols=self[1].ncols+2
    for i=1, #outs do
      mat[outs[i]+1][new_node_num]=in_weights[i] or 1
      mat[len+1][outs[i]+1]=out_weights[i] or 1
    end
    mat[len+1][new_node_num]=out_weights[#outs+1] or 1
    self.ninputs=self.ninputs+1
    self.nmemories=self.nmemories+1
    self.noutputs=self.noutputs+1
    self.size=self.size+1
    local keep_memories=t.keep_memories or t[6] or false
    if self.nmemories~=0 and keep_memories then
      self.memories=alg.New_Vector{alg.Vector_Iterator(self.memories), length=self.nmemories}--using function initializer for vectors with iterator over old memories to duplicate and fill new with 0s
    else
      self:Erase_Memories()
    end
  end
  self.size=self.size+1
  return self
end

function network.__index.Add_Link(self, t)
  -- call signature: Net:Add_Link{in_node(=), out_node(=) [, weight(=), override(=)]}
  --in_node must be provided as nonnegative integer less than the size of the network
  --out_node must be provided as nonnegative integer less than the size of the network
  --weight is optional and the default is 1
  --if a link already exists between in_node and out_node, override must be true else this operation will return false. In this scenario override true will replace the existing link weight with the given one
  --change_link depends on this
  local in_node=t.in_node or t[1]
  assert(type(in_node)=='number' and in_node>=0 and math.fmod(in_node,1)==0 and in_node<self.size,"Add Link Error: in_node must be provided as nonnegative integer less than the size of the network." )
  local out_node=t.out_node or t[2]
  assert(type(out_node)=='number' and out_node>=0 and math.fmod(out_node,1)==0 and out_node<self.size,"Add Link Error: out_node must be provided as nonnegative integer less than the size of the network." )
  local weight=t.weight or t[3] or  1
  assert(type(weight)=='number', "Add Link Error: weight must be number.")
  local mat=self[1][1]
  if mat[out_node][in_node]~=0 and not override then
    return false
  else
    mat[out_node][in_node]=weight
  end
  return self
end

function network.__index.Change_Link(self, t)
  --call signature: Net:Change_Link{in_node(=), out_node(=) [, weight(=)]}
  --in_node must be provided as nonnegative integer less than the size of the network
  --out_node must be provided as nonnegative integer less than the size of the network
  --weight is optional and the default is 1
  t.override=true
  self:Add_Link(t)
end

function network.Distance(A, B)
  --call signature: network.Distance(A, B)
  --calculates the distance between two networks A and B
  --distance method set in init.runtime.network_distance
  --choices of 'monte carlo' (default) TODO: finish 'Frobenius' (C function already written in basic_backend.c just needs avg function difference and fitness difference), monte carlo submethod: kullbach leibler divergence in case of softmax output, ...more options/combinations?
  --TODO: caching of some outputs and/or inputs to refine monte carlo approach
  --NOTE: RELATED TO THE TODO JUST ABOVE, MONTE CARLO APPROACH IS NOT A GOOD WAY TO WORK WHEN NETWORKS HAVE MEMORY. IT CANNOT TAKE THIS INTO ACCOUNT SINCE MEMORIES ARE TASK SPECIFIC GENERATED BY A NETWORK FOR ITSELF
  --Monte Carlo: random sample of inputs from given function (or default standard normal) some given number of iterations (default 1000)
  assert(A:Is_Network() and B:Is_Network(), "Network Distance Error: Both elements must be networks.")
  local method=init.runtime.network_distance
  if method[1]=='monte carlo' then
    local iters=method.iters or 1000
    local fun = method.fun or function() return math.sqrt(-2 * math.log(math.random())) * math.cos(2 * math.pi * math.random()) / 2 end, math.min(A.ninputs, B.ninputs)
    local x=nil
    local ret=0
    local diff=nil
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

function network.Mate(A, B)
end

function network.__index.Input(self, i)
  --call signature: Net:Input(i)
  --returns the index of structure belonging to the ith input (indexed from 0 & can return memory inputs without corresponding output)
  --errors if not a valid index
  assert(type(i)=='number' and math.fmod(i,1)==0 and i>=0 and i<self.ninputs, "Network Input Index Error: index not nonnegative integer or is out of range.")
  return i
end

function network.Input(self, i)
  --call signature: network.Input(Net, i)
  --returns the index of structure belonging to the ith input (indexed from 0 & can return memory inputs without corresponding output)
  --returns nil if not a valid index
  if i>=0 and i<self.ninputs then
    return i
  else
    return nil
  end
end



function network.__index.Hidden(self, i)
  --call signature: Net:Input(i)
  --returns the index of structure belonging to the ith hidden node (indexed from 0)
  --errors if not a valid index
  assert(type(i)=='number' and math.fmod(i,1)==0 and i>=0 and i<self.nhidden, "Network Hidden Index Error: index not nonnegative integer or is out of range.")
  return self.ninputs+i
end

function network.Hidden(self, i)
  --call signature: network.Hidden(Net, i)
  --returns the index of structure belonging to the ith hidden node (indexed from 0)
  --returns nil if not a valid index
  if i>=0 and i<self.nhidden then
    return self.ninputs+i
  else
    return nil
  end
end

function network.__index.Output(self, i)
  --call signature: Net:Input(i)
  --returns the index of structure belonging to the ith output (indexed from 0 & can return memory outputs without corresponding input)
  --errors if not a valid index
  assert(type(i)=='number' and math.fmod(i,1)==0 and i>=0 and i<self.noutputs, "Network Outputs Index Error: index not nonnegative integer or is out of range.")
  return self.ninputs+self.nhidden+i
end

function network.Output(self, i)
  --call signature: network.Output(Net, i)
  --returns the index of structure belonging to the ith output (indexed from 0 & can return memory outputs without corresponding input)
  --returns nil if not a valid index
  if i>=0 and i<self.noutputs then
    return self.ninputs+self.nhidden+i
  else
    return nil
  end
end

function network.__index.Memory(self, i)
  --call signature: Net:Input(i)
  --returns the indices of structure belonging to the ith memory's input/output pair (indexed from 0)
  --errors if not a valid index
  assert(type(i)=='number' and math.fmod(i,1)==0 and i>=0 and i<self.nmemories, "Network Memory Index Error: index not nonnegative integer or is out of range.")
  return self.ninputs-self.nmemories+i, self.size-self.nmemories+i
end

function network.Memory(self, i)
  --call signature: network.Memory(Net, i)
  --returns the indices of structure belonging to the ith memory's input/output pair (indexed from 0)
  --returns nil if not a valid index
  if i>=0 and i<self.nmemories then
    return self.ninputs-self.nmemories+i, self.size-self.nmemories+i
  else
    return nil
  end
end

function network.__sub(A, B)
  --call signature: A-B
  return network.Distance(A, B)
end

function network.__add(A, B)
  --call signature: A+B
  return network.Mate(A, B)
end

function network.__tostring(self)
  return tostring(self.structure)
end

setmetatable(network, network_mt)

return network
