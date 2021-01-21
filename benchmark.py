import torch
import numpy as np
import itertools
from time import time_ns
import matplotlib.pyplot as plt

# ==============================================================================
# Parameters
# ==============================================================================
# Note: We intend to perform a large grid search over many different parameters
#       The benchmark, however, uses only one set of parameters and varies the
#       number of data points (data_size) and system dimension (dim).
# Note: Number of epochs (epochs), usually denoting the number of training steps,
#       are ment to create some statistics of training/inference time here.
# Note: The results of this benchmark are stored in a folder ./results/ which is
#       not pushed into the git and must be created by hand!
# Note: The code works for both CPU and GPU which can be set with (device) and
#       (on gpu)
# Note: This file contains a minimal subset of the network idea and simplifies it
#       without loss of generality in respect to the benchmark

PARAMS = {
# Size of the network
    "nll": 1, # number of linear layers composing the affine members
    "ncl": 3, # number of affince compling layers, must be larger then 3!
    "dim": [48,60,72,84,96], # dimension of system, for our cns we have dim = N_sigma^d * N_t
# Size of the data;
    "data size": [1_000,2_000,3_000,4_000,5_000,6_000,7_000,8_000,9_000,10_000], # size of the training data i.e. train_dat.size() = (PARAMS["train size"],PARAMS["dim"])
# Training parameters
    "lr": 1e-2,
    "wd": 0.8,
    "epochs": 100,
# Compute information
    "device": torch.device("cuda"), # On which device (cuda,cpu) to execute
    "on gpu": True,                 # Set to false if device is cpu
    }

# security check
if PARAMS["on gpu"]:
    if not torch.cuda.is_available():
        raise RuntimeError(f"No GPU device found. Is you torch installation correct?")

# ==============================================================================
# Model Definition
# ==============================================================================
class RandomAffineCouplingLayer(torch.nn.Module):
    def __init__(self,dim,*args,**kwargs):
        r"""
            dim: int
                dimension of the physical system
            *args:
                Arguments passed to the torch.nn.Module class
            *kwargs:
                Keyworded arguments passed to the torch.nn.Module class

            The Random Affine Coupling Layer draws two integer partitions A,B of
            same size such that (A U B) equals the index interval [1,dim].
            Then the input vector, x, can be partitioned into two subsets
            x_A,x_B. With that one defines the Affine Coupling Layer as
                       | x_A
                f(x) = |
                       | f(x_B) * x_A + g(x_B)
            where f,g are arbitrary functions. In prticular, we apply all sorts
            of neuronal networks as f,g (Sequence of Linear layers here). f,g are
            called affine members here.
        """
        super(RandomAffineCouplingLayer,self).__init__(*args,**kwargs)

        # Create the affine members (am): f(x) = am_mul(x) + am_add
        # Note: For simplicity these are stacks of linear layers our application
        #       has a possibility to change the network architecture in here.
        self.am_mul = torch.nn.Sequential( *([torch.nn.Linear(dim,dim), torch.nn.ReLU()]*PARAMS["nll"]) )
        self.am_add = torch.nn.Sequential( *([torch.nn.Linear(dim,dim), torch.nn.ReLU()]*PARAMS["nll"]) )

        # create a mask making up which input elements are copied and which are
        # passed to the affine transformation
        self.mask_A = torch.zeros(size=(dim,), dtype=torch.uint8)
        self.mask_B = torch.ones(size=(dim,), dtype=torch.uint8)
        # Randomly draw which element indices, we do not allow for doubling here!
        self.indices = torch.randperm(dim)[::2]
        self.mask_A[self.indices] = 1 # True
        self.mask_B[self.indices] = 0 # False
        # register the masks as parameter
        if PARAMS["on gpu"]:
            self.mask_A = torch.nn.parameter.Parameter( self.mask_A.to(PARAMS["device"]),requires_grad=False )
            self.mask_B = torch.nn.parameter.Parameter( self.mask_B.to(PARAMS["device"]),requires_grad=False )
        else:
            self.mask_A = torch.nn.parameter.Parameter( self.mask_A,requires_grad=False )
            self.mask_B = torch.nn.parameter.Parameter( self.mask_B,requires_grad=False )

    def forward(self,input):
        return (input*self.mask_A) * self.am_mul(input*self.mask_B) \
                                   + self.am_add(input*self.mask_B)

# define the loss function
loss_function = torch.nn.L1Loss()

def benchmark(data_size,dim):
    r"""
        data_size: int
            Number of training data points
        dim: int
            Dimension of the system

        This Benchmark function mimics the training and inference (validation) of
        the Random Affine Coupling Layer define above. The training time per epoch
        and validation time in each epoch is measured and averaged.
        Intend to call this function for the different data_sizes and dimensions
        we want to include into the benchmark.
    """
    # stacking up ncl affine coupling layers
    if PARAMS["on gpu"]:
        model = torch.nn.Sequential( *([RandomAffineCouplingLayer(dim)]*PARAMS["ncl"]) ).to(PARAMS["device"])
    else:
        model = torch.nn.Sequential( *([RandomAffineCouplingLayer(dim)]*PARAMS["ncl"]) )

    # define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=PARAMS["lr"],weight_decay=PARAMS["wd"])

    # create data
    rdat = (torch.rand((data_size,dim),device = PARAMS["device"]),torch.rand((data_size,dim),device = PARAMS["device"]))

    # put the data into the torch interface
    train_dat = torch.utils.data.TensorDataset(*rdat)
    train_datLoader = torch.utils.data.DataLoader(train_dat,batch_size=data_size)

    # storage for the timing data
    bench_epoch_infer = np.zeros(shape=(PARAMS["epochs"],))
    bench_epoch_train = np.zeros(shape=(PARAMS["epochs"],))

    # Note: The epochs here are just a tool to generate statistics of the timing
    #       Correctness of the model is not tested in this benchmark
    for epoch in range(PARAMS["epochs"]):
        s_epoch = time_ns() * 1e-9
        # # there is only one minibatch
        for data,label in train_datLoader:
            # make a prediction
            pred = model(data)
            # compute loss
            loss = loss_function(pred,label)
            # zero the gradient data
            optimizer.zero_grad()
            # backpropagate
            loss.backward()
            # update parameters
            optimizer.step()
        e_epoch = time_ns() * 1e-9

        # measure inference under no_grad condition!
        with torch.no_grad():
            for data,label in train_datLoader:
                # no minibatches here, thus this loop has only one execution
                s_infer = time_ns() * 1e-9
                pred = model(data)
                e_infer = time_ns() * 1e-9

        # store time in seconds
        bench_epoch_infer[epoch] = e_infer - s_infer
        bench_epoch_train[epoch] = e_epoch - s_epoch

    infer_time_per_epoch = np.average(bench_epoch_infer)
    train_time_per_epoch = np.average(bench_epoch_train)

    return  infer_time_per_epoch,train_time_per_epoch

# create arrays to store the train/inference time
timing_train = np.zeros(shape=( len(PARAMS['dim']),len(PARAMS['data size'])) )
timing_infer = np.zeros(shape=( len(PARAMS['dim']),len(PARAMS['data size'])) )

# create a figure to plot the benchmark results
fig = plt.figure(figsize=(8.0,5.85)) #(width,height)
train_ax = fig.add_subplot(1,2,1)
infer_ax = fig.add_subplot(1,2,2)

# benchmark the network for the given data sizes and dimension
for i_d,dim in enumerate(PARAMS["dim"]):
    print(f"Benchmarking dimension {dim}")
    for i_ds,data_size in enumerate(PARAMS["data size"]):
        timing_infer[i_d,i_ds],timing_train[i_d,i_ds] = benchmark(data_size,dim)
    # plot the results
    train_ax.plot(PARAMS["data size"],timing_train[i_d,:]*1e+3,'.-',label=f"System dimension = {dim}")
    infer_ax.plot(PARAMS["data size"],timing_infer[i_d,:]*1e+3,'.-',label=f"System dimension = {dim}")

# postprocess the plot
if PARAMS["on gpu"]:
    infer_ax.set_title("GPU Benchmark: Inference Time per Epoch")
    train_ax.set_title("GPU Benchmark: Training Time per Epoch")
else:
    train_ax.set_title("CPU Benchmark: Training Time per Epoch")
    infer_ax.set_title("CPU Benchmark: Inference Time per Epoch")

train_ax.set_xlabel("Training data size")
infer_ax.set_xlabel("Training data size")

# only on the left plot
train_ax.set_ylabel("Average Execution Time per Epoch [ms]")

train_ax.grid()
infer_ax.grid()

handles, labels = train_ax.get_legend_handles_labels()

fig.legend(handles,labels,bbox_to_anchor=(0.7, 0.25),
          ncol=1, fancybox=True, shadow=True, fontsize='large')

fig.tight_layout(rect=[0,0.25,1,1])

if PARAMS["on gpu"]:
    fn = "results/GPU"
else:
    fn = "results/CPU"

fn+=f"Bench_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.pdf"

plt.savefig(fn)
