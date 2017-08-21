function love.load()
end

local init=require "init"
local alg=require "alg.alg_basic"
local inspect=require "inspect"

local ffi=require "ffi"

local function print(obj)
  love.graphics.print(inspect(obj))
end

local n=alg.New_Function{{math.sin,math.cos,math.tanh}, 10}


local m=alg.New_Matrix{{{1,2},{3,4}}, 5,4}
m:Values()[0][7]=.5
--local a=alg.Square_Matrix_Matrix{m, n}
function love.draw()
  --print(inspect{alg.Vector_Length(u),alg.Vector_Length(v)})
  --print(alg.Vector_To_String(alg.backend.Vector_Vector_Add(u,v,alg.Vector_Length(u),alg.Vector_Length(v),ret)))
  --love.graphics.print(alg.Matrix_To_String(ans))
  love.graphics.print(jit.version)
  --print(ffi.cast('void *',ffi.new('float[10]')))
end
