function love.load()
end

local init=require "init"
local alg=require "alg.alg_basic"
local inspect=require "inspect"
local functions, my_functions=require("alg.functions")()
local ffi=require "ffi"

local function print(obj)
  love.graphics.print(inspect(obj))
end

local n=alg.New_Function{{math.sin,math.cos,math.tanh}, 10}


local m=alg.New_Matrix{{{1,2},{3,4}}, 5,4}
m:Values()[0][7]=.5
--local a=alg.Square_Matrix_Matrix{m, n}

local alpha=.9
local temp=function(x) if x<0 then return alpha*(math.exp(x)-1) else return x end end
local elu1=functions.New{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu1'}
local elu2=my_functions{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu2'}

function love.draw()
  --print(inspect{alg.Vector_Length(u),alg.Vector_Length(v)})
  --print(alg.Vector_To_String(alg.backend.Vector_Vector_Add(u,v,alg.Vector_Length(u),alg.Vector_Length(v),ret)))
  --love.graphics.print(alg.Matrix_To_String(ans))

  --print(ffi.cast('void *',ffi.new('float[10]')))
end
