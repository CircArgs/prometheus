--[[
TODO: if derivative not given then approximate with finite difference and warn of severe performance impact?

A function library for use with alg
-----------------------------------------------------------------------------------------------------
a function is a ctype typdef'd below
-----------------------------------------------------------------------------------------------------
can add, subtract, multiply and divide functions
-----------------------------------------------------------------------------------------------------
can create new named functions (stored in User because cannot directly modify functions as it's a metatype of typedef 'functions' AND this makes it impossible to overwrite default functions without a loss of functionality)
-----------------------------------------------------------------------------------------------------
comes with some default functions non of which have shape and scale paramters. Functions with shape and scale parameters can be declared separately (or specific constructor methods could be written which require these parameters to be defined before use)
-----------------------------------------------------------------------------------------------------
the library must be imported in one of two ways: i) local name_you_want=require "alg.functions" () --this will bring in the library and user defined functions will be stored in functions.User (see more details below)
or ii) local name_you_want_for_default, name_you_want_for_your_own_functions=require "alg.functions"()
-----------------------------------------------------------------------------------------------------
user defined functions can be defined in one of the following 3 ways:
name_you_want_for_your_own_functions{func(=),deriv(=),name(=)}
name_you_want_for_default.New{func(=),deriv(=),name(=)}
name_you_want_for_default.User{func(=),deriv(=),name(=)}

OR

you can use anything of C Type function (anything functions.Is_Function() returns true for)
with analagous syntax to the above but ommitting the deriv
ex. name_you_want_for_default.New{func(=),name(=)}


user defined functions can be accessed in any of the following ways:
name_you_want_for_your_own_functions.name
name_you_want_for_default.User.name
name_you_want_for_default.New.name
-----------------------------------------------------------------------------------------------------
you can get a table of the defined function names with:
functions.List_Functions() or functions.List_Functions('table')

alternatively, you can get the same result in string for via
functions.List_Functions('string')

alternatively, you can call the library itself to get the same respective results:
functions('table') for the table form and functions('string')

ALSO you can see if a function is defined with a particular name is defined via:
functions('name_of_function')

which will return true if the function is defined as a default and true, '[USER]' if the function is defined already by the User
-----------------------------------------------------------------------------------------------------
[if you stored a created function in a variable, naturally you can access it with that variable]
ex: This example demonstrates one way one might make a parameterized activation function in this case the elu activation with parameter alpha=.9
--Note you can define new functions using your own functions, functions from this library, or any function (including C functions)
-----------------------------------------------------------------------------------------------------
local functions, my_functions=require('alg.functions')()

local alpha=.9
local temp=function(x) if x<0 then return alpha*(math.exp(x)-1) else return x end end
local elu1=functions.New{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu1'}
local elu2=my_functions{temp, function(x) if x<0 then return temp(x)+alpha else return 1 end end, 'elu2'}

print(elu1==my_functions.elu1 and elu2==my_functions.elu2) -->true (note that elu1 is not equal to elu2 as these are pointers and the functions have different locations in memory)
print(my_functions.elu1(10))-->10
print(elu2.derivative(-1000)==elu2:Derivative(-1000))-->true

]]

local init=require("init")
local ffi=require("ffi")
local TYPE=init.TYPE

ffi.cdef(
"typedef "..TYPE.." TYPE;"..[[
typedef TYPE (*func_ptr)(TYPE);
typedef struct {
  func_ptr func;
  func_ptr derivative;
} function;
]])

local User={defined_string='', defined_table={}}--table for user defined functions after the module is imported
local User_mt={}--metatable for the table of user defined table

local functions={User=User, New=User, __index={}}
local functions_mt={}

function functions.__call(self, x)
  return self.func(x)
end

function functions.__index.Is_Function(self)
  return ffi.istype('function', self)
end

function functions.__index.Derivative(self, x)
  if type(x)=='number' then
    return self.derivative(x)
  else
    return self.derivative
  end
end

function functions.__index.Func(self,x)
  --call signature: self:Func(3) returns value of function evaluated at 3; self:Func() returns pointer to function
  if type(x)=='number' then
    return self.func(x)
  else
    return self.func
  end
end

local function defined(name)
  if #User.defined_string==0 then
    User.defined_string=User.defined_string..name
  else
    User.defined_string=User.defined_string..', '..name
  end
  User.defined_table[#User.defined_table+1]=name
end

function User_mt.__call(self, t)
  --adds a new function to User (or name of User's table of self-defined functions upon library import)
  --returns the created function so it may be stored in a variable
  local func=t[1] or t.func
  local derivative=t[2] or t.derivative
  local name=t[3] or t.name
  if ffi.istype('function',func) then
    if type(t[2])=='string' then
      name=name or t[2]
      self[name]=func
      defined('[USER] '..name)
      return self[name]
    end
    func=func.func
  end
  if ffi.istype('function',derivative) then
    derivative=derivative.derivative
  end
  --assert(type(func)=='function' and type(derivative)=='function', 'New Function Instantiation Error: Given values must be functions.')
  assert(type(name)=='string', "New Function Instantiation Error: Name must be string.")
  self[name]=ffi.new('function', {func, derivative})
  defined('[USER] '..name)
  return self[name]
end

function functions.__add(f1, f2)
  return ffi.new('function',function(x) return f1.func(x)+f2.func(x) end,function(x) return f1.derivative(x)+f2.derivative(x) end)
end

function functions.__sub(f1, f2)
  return ffi.new('function',function(x) return f1.func(x)-f2.func(x) end,function(x) return f1.derivative(x)-f2.derivative(x) end)
end

function functions.__mul(f1, f2)
  return ffi.new('function',function(x) return f1.func(x)*f2.func(x) end,function(x) return f1.derivative(x)*f2.func(x)+f1.func(x)*f2.derivative(x) end)
end

function functions.__div(f1, f2)
  return ffi.new('function',function(x) return f1.func(x)/f2.func(x) end,function(x) return (f1.derivative(x)*f2.func(x)-f1.func(x)*f2.derivative(x))/f2.func(x)^2 end)
end

function functions.List_Functions(o)
  --function to tell what functions have been declared.
  --returns table by default
  assert(o==nil or o=='string' or o=='table', "List_Functions Error: Option is need be either nil, 'string' or 'table'.")
  local o=o or 'table'
  return User['defined_'..o]
end

function functions_mt.__call(self, s)
  assert(type(s)=='string', "Functions Check/List Error: Argument must be string.")
  if s~='string' and s~='table' then
    for _,v in ipairs(self.New.defined_table) do
      if s==v then
        return true
      elseif '[USER] '..s==v then
        return true, '[USER]'
      end
    end
    return false
  else
    return self.List_Functions(s)
  end
end
--be sure to add default functions' names to the list of those defined
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
functions.sin=ffi.new('function', {math.sin, math.cos})
defined'sin'
functions.cos=ffi.new('function', {math.cos, function(x) return -math.cos end})
defined'cos'
functions.logistic=ffi.new('function', {function(x) return 1/(1+math.exp(-x)) end, function(x) return -math.exp(x)/(1+math.exp(x))^2 end})
defined'logistic'
functions.softplus=ffi.new('function', {function(x) return math.log(math.exp(x)+1) end, functions.logistic.func})
defined'softplus'
functions.relu=ffi.new('function', {function(x) return math.max(x,0) end, function(x) if x>0 then return 1 else return 0 end end})
defined'relu'
functions.tanh=ffi.new('function', {math.tanh, function(x) return 1-math.tanh(x)^2 end})
defined'tanh'
functions.gaussian=ffi.new('function', {function(x) return math.exp(-x^2) end, function(x) return -2*x*math.exp(-x^2) end})
defined'gaussian'
functions.linear=ffi.new('function', {function(x) return x end,function (x) return 1 end})
defined'linear'
functions.arctan=ffi.new('function', {math.atan, function(x) return 1/(x^2+1) end})
defined'arctan'
functions.softsign=ffi.new('function',{function(x) return x/(1+math.math.abs(x)) end, function(x) return 1/(1+math.math.abs(x))^2 end})
defined'softsign'
functions.binary_step=ffi.new('function', {function(x) return x>0 end, function(x) return 0 end})
defined'binary_step'
functions.tanh=ffi.new('function', {math.tanh, function(x) return 1/(1+x^2) end})
defined'tanh'
functions.sinc=ffi.new('function', {function(x) if x~=0 then return math.sin(x)/x else return 1 end end, function(x) if x~=0 then return (math.cos(x)-math.sin(x)/x)/x else return 0 end end })
defined'sinc'
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
setmetatable(User, User_mt)
setmetatable(functions, functions_mt) --you can call the library to the same tune as functions.List_Functions(s)
ffi.metatype('function', functions)

return function() return functions, User end
