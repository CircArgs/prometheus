local TYPE="double" --define the type for computations
local root=love.filesystem.getSourceBaseDirectory()


return {
runtime={
initial_pop_size=10,
max_pop_size=math.huge,--the maximum size the population will grow to. If this size is hit, the weakest members of the population will all be killed off each epoch. math.huge=infinity
add_node_p=.1,
memory_node_p=.1,--probability given a node is added that it is a memory node (new input-output pair)
change_node_p=.05,
add_link_p=.1,
change_link_p=.1,
spontaneous_combustion=.0001,--chance a member of the population will randomly die
network_distance={'monte carlo'},--network distance must be a table with first element the method name and remaining elements named options [currently supports: 'monte carlo', ]
memory_activation={'sigmoid'}, --table of output activations for memory nodes to be chosen from randomly
mating_method={'average', type='fitness'} --[[method that will be used when mating two networks. average [types: fitness- weighted average proportional to fitness]: takes a weighted average of the activations and link weights
]]
},
version='0.0.1',
basic_path="C:/Prometheus/Prometheus.love/Basic",
blas_path=root.."/Prometheus.love/OpenBLAS",
opencl_path=root.."/Prometheus.love/OpenCL",
root=root,
TYPE=TYPE
}
