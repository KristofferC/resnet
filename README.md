#### Benchmark Resnet50 on julia (Flux) and Python (Pytorch)

##### Download dataset:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200
```

#### Run julia:

```
julia --project -e 'import Pkg; Pkg.instantiate()'

julia --project resnet_julia_bench.jl
```

