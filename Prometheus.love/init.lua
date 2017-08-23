local TYPE="double" --define the type for computations
local root=love.filesystem.getSourceBaseDirectory()


return {
version='0.0.1',
basic_path="C:/Prometheus/Prometheus.love/Basic",
blas_path=root.."/Prometheus.love/OpenBLAS",
opencl_path=root.."/Prometheus.love/OpenCL",
root=root,
TYPE=TYPE
}
