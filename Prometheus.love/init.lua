local TYPE="double" --define the type for computations
local root=love.filesystem.getSourceBaseDirectory()


return {
runtime={
initial_pop_size=10,
max_pop_size=math.huge,--the maximum size the population will grow to. If this size is hit, the weakest members of the population will all be killed off each epoch. math.huge=infinity
add_node_p=.1,
change_node_p=.05,
add_link_p=.1,
change_link_p=.1,
spontaneous_combustion=.0001,--chance a member of the population will randomly die
network_distance={'monte carlo'}--network distance must be a table with first element the method name and remaining elements named options


},
version='0.0.1',
basic_path="C:/Prometheus/Prometheus.love/Basic",
blas_path=root.."/Prometheus.love/OpenBLAS",
opencl_path=root.."/Prometheus.love/OpenCL",
root=root,
TYPE=TYPE
}
