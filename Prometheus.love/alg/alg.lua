--[[this script serves to determine what system will be used to perform the underlying computations.
--currently does not have gpu acceleration so all computations use the came backend
TODO: add opencl backend (possibly add cuda backend too?) as high backends (i.e. backends for especially large computations for which the speed bonus of the gpu acceleration outweighs the memory latency of copying to gpu memory)
Batch compile OpenBlas on install for system specific optimizations
]]

local init = require("init")
local templet = require("templet")
local ffi=require("ffi")
local root=init.root
package.path=package.path..";"..root.."?.lua"
local os_name=ffi.os

--OpenBlas
local try_openblas,catch_openblas=pcall()
local try_blas_cores, catch_blas_cores=pcall(love.system.getProcessorCount)--try to get the core count (thread count really i.e. on a hyperthreaded Intel cpu w/ 4 cores this will yield 8)
local BLAS_CORES=1
if try_blas_cores and tonumber(catch_blas_cores)>2 then
  BLAS_CORES = tonumber(catch_blas_cores)-2 --reserve 2 cores for system tasks and host processes (i.e. OS and kernel dispatching respectively)
end

--[[Using OpenBlas backend]]
local OpenBlas = templet.loadstring[[
local ffi = require("ffi")
local openblas=require(local compute=require(${root}.."OpenBlas//${os}//x64//libopenblas)
local backend=${backend}
local num_threads=${num_threads}
local cblas_h = require(root.."OpenBlas//cblas_h")--"header" file for C environment
ffi.cdef(cblas_h)

function Matrix_Matrix_Elwise(left,right)
  return
end

function Vector_Vector_Elwise(left,right)
  return
end

function Matrix_Matrix(left,right)
  return
end
]]

return OpenBlas{os=os_name, backend='openblas', num_threads=BLAS_CORES, root=root}

local basic_backend, catch=require(root.."basic_backend")
assert(basic_backend, "Critial Error: No backend found.")

local Basic = templet.loadstring([[
local ffi = require("ffi")
local openblas=require(local compute=require(${root}.."Basic//basic_backend)
local backend='basic'
local num_threads=${num_threads}
local cblas_h = require(root.."OpenBlas//cblas_h")--"header" file for C environment
ffi.cdef(cblas_h)

function Matrix_Matrix_Elwise(left,right)
  return nil
end

function Vector_Vector_Elwise(left,right)
  return nil
end

function Matrix_Matrix(left,right)
  return nil
end

function Vector_Vector(left,right)
  return Vector_Vector(left,)
]])
return Basic{os=os_name, num_threads=1, root=root}--[[Using basic C backend]]
