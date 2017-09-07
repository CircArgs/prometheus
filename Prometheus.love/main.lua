function love.load()
end

local init=require "init"
local version=init.version
local alg=require "alg.alg_basic"
local inspect=require "inspect"
local functions, my_functions=require("alg.functions")()
local ffi=require "ffi"
local networks=require"network"
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--These are just printing function wrappers for printing to debug. the love print function does not implicitly cast to string so print3 is like a regular print function
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
local function print1(obj)
  love.graphics.print(inspect(obj))
end

local function print2(obj)
  love.graphics.print(obj)
end

local function print3(obj)
  love.graphics.print(tostring(obj))
end
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--just demonstrating some different syntax for defining User created functions (things can get confusing so best choose a syntax and stick with it. Just demoing the versatility and freedom)
functions.User{functions.linear, 'g1'} --a simple renaming of a default function (the original remains. Had we used a name already in use for a user defined function the user defined function would be overwritten. Overwriting the default functions is not possible nor is ambiguity possible)
my_functions{function(x) return x^2 end, function(x) return 2*x end, 'g2'}
local g3=functions.New{math.sqrt, function(x) return .5/math.sqrt(x) end, 'g3'}

local two=functions.User{function(x) return 2 end, function(x) return 0 end, 'two'}--we can store the function in a local variable (it will still be stored in the typical table also)
functions.User{function(x) return 3 end, function(x) return 0 end, 'three'}
functions.New{two*functions.linear, 'g4'}--we can perform algebra with the functions to make new functions!!!
local g5=my_functions{my_functions.three*functions.User.g2, 'g5'}

local F=alg.New_Function{{functions.User.g1, functions.New.g2, g3, my_functions.g4, g5}}

local structure=alg{new='matrix', {{0,0,0,0,0},{.1,0,0,0,0},{.25,2.9,1,0,0},{.3,3.14,0,1,0},{.5,41.2,0,0,1}}}--just one way to make a matrix. There's a much lower level way and another utility function alg.New_Matrix
local A=networks{structure, F, ninputs=1, noutputs=3, nhidden=1}--you can make the first arguments named as well (see network class for details)

--[[--just a demo for showing how some fancier functions that want further parameters can be defined
local alpha=1
local temp=function(x) if x<0 then return alpha*(math.exp(x)-1) else return x end end
local elu1=functions.New{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu1'}
local elu2=my_functions{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu2'}
]]

--[[--here we have the computation that goes on when A(X) is called written out completely
X=alg.Square_Matrix_Vector{A[1], X}--if we could we'd overwrite X to save time managing memory, but with matrix multiplication you need to keep values on hand until the end of the calculation so here we just reassign the pointer to X and let the old memory of X get picked up by GC
alg.Vector_Function{F, X, A.ninputs, A.ninputs+A.nhidden-1, overwrite=true}--overwite just means that we get rid of the values of X during the operation
X=alg.Square_Matrix_Vector{A[1], X}
alg.Vector_Function{F, X, overwrite=true}
]]
A:Add_Node{nil, {A:Hidden(0)}, in_weights={.5}, out_weights={1.377, .3}}--adds a new memory node connected to the first (at this line the only) hidden node
A.memories[1][0]=.15651
A:Add_Node{keep_memories=true}
A:Add_Node{{A:Input(0)}, {A:Hidden(0), A:Output(2)}, functions.relu, {.314}, {.367, .52}}--add a node that is connected to the first input, first hidden and last ouput with weights given (so won't delete connection between first input and current hidden node)
--an empty call sees ins as nil and will simply make a memory in/out pair with connection weight 1
--A:Add_Link{in_node=A:Input(0), out_node=A:Hidden(0), weight=.5} --add a link from the the input of the first memory to the first hidden node with weight .5 (named arguments optional)
--local mem1in, mem1out=A:Memory(0) --get the indices of the in/out pair for the first memory node

--local X=alg{new='vector', function() return 1 end, length=A.ninputs}--X is to be the input. We are going to feed 1 to all the inputs of network A. demonstrating vectors can be generated wil functions (functions can take index input to generate vector values based on position)
local X=alg{new='vector', {1}}--an identical (cleaner in this case) way to do what's just above

--[[
c=alg.Square_Matrix_Vector{A.structure, X}
alg.Vector_Function{F, c, A.ninputs, A.ninputs+A.nhidden-1, overwrite=true}
c=alg.Square_Matrix_Vector{A.structure, c}
alg.Vector_Function{F, c, A.ninputs, A.ninputs+A.nhidden-1, overwrite=true}
c=alg.Square_Matrix_Vector{A.structure, c}
alg.Vector_Function{F, c, overwrite=true}]]

--local output=A(X)--simple syntax. A is the network, X is the input. Call A like the function it is (using metamethod __call to intepret A as a function)

c=A(X)
function love.draw()
  --print3(P)--uncomment to see the adjacency matrix
  --print3(X)--What an input in this example looks like
  --print3(mem1in..', '..mem1out..'\n')
  --print3(A.activations[1][A:Hidden(1)].derivative(120561))--new hidden relu node has in fact been added and we can see it's proper derivative evaluated
  --print3(A.size)
  print3(c)
end
