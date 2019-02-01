#### Benchmark Resnet50 on julia (Flux) and Python (Pytorch)

##### Download dataset:

```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

#### Run julia:

```
julia --project -e "import Pkg; Pkg.instantiate()"

julia --project resnet_julia_bench.jl
```

Only doing 30 batches

```
Epoch: 1, Batch 1 / 30, 0.61 Images / sec
Epoch: 1, Batch 2 / 30, 10.77 Images / sec
Epoch: 1, Batch 3 / 30, 8.28 Images / sec
Epoch: 1, Batch 4 / 30, 8.39 Images / sec
Epoch: 1, Batch 5 / 30, 9.60 Images / sec
Epoch: 1, Batch 6 / 30, 9.77 Images / sec
Epoch: 1, Batch 7 / 30, 10.04 Images / sec
Epoch: 1, Batch 8 / 30, 9.83 Images / sec
Epoch: 1, Batch 9 / 30, 11.93 Images / sec
Epoch: 1, Batch 10 / 30, 11.91 Images / sec
```

#### Run Python

Have pytorch + pytorchvision installed

```
python resnet.py -a resnet50 tiny-imagenet-200 --batch-size=64
```

```
Epoch: [0][0/1563]      Time 5.711 (5.711)      Images /sec 11.207      Data 1.271 (1.271)      Loss 7.1003 (7.1003)    Acc@1 0.000 (0.000)     Acc@5 0.000 (0.000)
Epoch: [0][10/1563]     Time 0.258 (0.769)      Images /sec 247.765     Data 0.000 (0.116)      Loss 15.0394 (11.8107)  Acc@1 0.000 (0.426)     Acc@5 3.125 (2.415)
Epoch: [0][20/1563]     Time 0.255 (0.525)      Images /sec 250.656     Data 0.000 (0.068)      Loss 7.6263 (10.0656)   Acc@1 0.000 (0.521)     Acc@5 3.125 (2.902)
Epoch: [0][30/1563]     Time 0.257 (0.440)      Images /sec 248.725     Data 0.001 (0.051)      Loss 7.0500 (9.1263)    Acc@1 0.000 (0.504)     Acc@5 6.250 (2.772)
```
