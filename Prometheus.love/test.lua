local ffi = require("ffi")

local init = require("init")

ffi.load("basic_backend")
local basic_backend= require("basic_backend_h")

local v=ffi.new('float[?]',100)
v[0]=.5
return {
a=basic_backend.Vector_Length(v)






}
