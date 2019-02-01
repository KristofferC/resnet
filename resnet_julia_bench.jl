import Pkg
Pkg.activate(@__DIR__)

using Metalhead
using CSV
using DataFrames
using Flux
using CuArrays

using Random
using Printf

const DATASET_PATH = joinpath(@__DIR__, "tiny-imagenet-200")
const CATEGORIES = sort(readdir("$DATASET_PATH/train"))

const MODEL_GPU = Metalhead.resnet50();

# We are only using 200 categories for this experiment so swap out the last FC layer
const MODEL_GPU_200 = Chain(
    ntuple(i -> (i != (length(MODEL_GPU.layers)-1) ? MODEL_GPU.layers[i] : Dense(2048, length(CATEGORIES))), length(MODEL_GPU.layers))...
) |> gpu


const LABEL_TO_TEXT = Dict{String, String}()
let
for row in CSV.File("$DATASET_PATH/words.txt"; delim='\t', header=[:label, :description])
    LABEL_TO_TEXT[row.label] = row.description
end
end

categories = readdir("$DATASET_PATH/train")
const TRAIN_DATA = Tuple{String, String}[]
for category in categories
    path = "$DATASET_PATH/train/$category/images"
    for image in readdir(path)
        push!(TRAIN_DATA, ("$path/$image", category))
    end
end
shuffle!(TRAIN_DATA);

const VALIDATION_DATA_LABELS = Dict{String, String}()
const VALIDATION_DATA = Tuple{String, String}[]
let
for row in CSV.File("$DATASET_PATH/val/val_annotations.txt"; delim='\t', header=[:file, :label, :x0, :y0, :x1, :x2])
    VALIDATION_DATA_LABELS[row.file] = row.label
end
for image in readdir("$DATASET_PATH/val/images")
    f = "$DATASET_PATH/val/images/$image"
    if isfile(f)
        push!(VALIDATION_DATA, (f, VALIDATION_DATA_LABELS[image]))
    end
end
end

function prepare_batch(data, batch_number, batch_size)
#    println("Preparing batch $batch_number")
    start = batch_size * (batch_number - 1) + 1
    batch_size = min(batch_size, length(data) - start + 1)
    img_1 = Metalhead.preprocess(data[start][1])
    x = zeros(Float32, (size.((img_1,), (1,2,3))..., batch_size))
    x[:,:,:, 1] = img_1
    for (i, idx) in enumerate(start+1:start+batch_size-1)
        x[:,:,:,i+1] = Metalhead.preprocess(data[idx][1])
    end
    y = Flux.onehotbatch([data[i][2] for i in start:start+batch_size-1], CATEGORIES)
    #println("Done with batch")
    return (x, y)
end

function loss(x, y)
    Flux.crossentropy(MODEL_GPU_200(x), y)
end

accuracy(x, y) = mean(Flux.onecold(MODEL_GPU(x)) .== Flux.onecold(y))



opt = ADAM()

BATCH_SIZE = 16
# N_BATCHES = length(TRAIN_DATA) ÷ BATCH_SIZE
N_BATCHES = 30

# Async feed data to GPU
c = Channel(6)
@async begin
    for i in 1:N_BATCHES
        put!(c, gpu(prepare_batch(TRAIN_DATA, i, BATCH_SIZE)))
    end
    close(c)
end

N_EPOCHS = 1

for epoch in 1:N_EPOCHS
    prev_time = time()
    for (i, (x, y)) in enumerate(c)

      l = loss(x, y)
      Flux.back!(l)
      Flux.Optimise._update_params!(opt, Flux.params(MODEL_GPU_200))
      t = time()
      Δt = t - prev_time
      @printf "Epoch: %d, Batch %d / %d, %4.2f Images / sec \n" epoch i N_BATCHES size(x, 4) / Δt
      prev_time = t
    end
end
