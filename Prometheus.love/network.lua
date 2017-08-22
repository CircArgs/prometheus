--[[
class for networks
TODO: currently can only call a network to evaluate on a vector. need to add matrices (larger batched than 1 i.e. a single vector)
]]

local init=require "init"
local alg=require "alg.alg_basic"

local network={__index={}}

function network.__Call(self, v, grad)
  assert(v:Is_Vector(), "Network Evaluation Error: Networks can currently only be evaluated on vectors.")
  local F=self.F
  local A=self[1][1]--[1] get matrix, [1][1] get values from matrix
  local v1=v[1]
  if not grad then
    for i=1, self.nhidden do
      alg.Square_Matrix_Vector{A, ret, overwrite=true}
      alg.Function_Vector{F, v1, self.ninputs, self.ninputs+self.nhidden-1, overwrite=true}
    end
    alg.Function_Vector{F, v1, overwrite=true}
    return v
  else
    local dA=self:dA()
    for i=1, self.nhidden do
      alg.Square_Matrix_Vector{A, ret, overwrite=true}
      alg.Function_Vector{F, v1, self.ninputs, self.ninputs+self.nhidden-1, overwrite=true}
      alg.
    end
    alg.Function_Vector{F, v1, overwrite=true}
    return {v, {dA, nrows=A.nrows, ncols=A.ncols, trans=false}}
  end
end

function alg.__index.dA(self)
  local len=self[1].nrows
  assert(len=self[1].ncols,"Matrix Dualpart Fetch Error: Matrix must be square.")
  local dA=ffi.new('TYPE ['..len..']['..len..']')
  alg.backend.Square_dA(self[1][1], len, dA)
  return dA
end

function network.New_Network(t)
  local
