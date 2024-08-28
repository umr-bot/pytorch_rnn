export USE_NUMPY=1
export USE_NINJA=OFF
export CMAKE_CUDA_COMPILER=/usr/local/cuda-11.4/bin/nvcc
export CUDACXX=/usr/local/cuda-11.4/bin/nvcc     11:01:15
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
export TORCH_CUDA_ARCH_LIST="3.5"  
export LDFLAGS=-fno-lto
#export BUILD_CAFFE2=0; export BUILD_CAFFE2_OPS=0 # disble caffe
