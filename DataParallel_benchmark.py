import torch
import pandas as pd
import itertools
from time import time,time_ns
import matplotlib.pyplot as plt

PARAMS = {
    "nll": 1, # number of linear layers composing the affine members
    "ncl": 3, # number of affince compling layers, must be larger then 3!
    "dim": [48,60,72,84,96], # dimension of system, for our cns we have dim = N_sigma^d * N_t
# Size of the data;
    "data size": [10_000,20_000,30_000,40_000,50_000,60_000,70_000,80_000,90_000,100_000], # size of the training data i.e. train_dat.size() = (PARAMS["train size"],PARAMS["dim"])
# Training parameters
    "lr": 1e-2,
    "wd": 0.8,
    "epochs": 10,
# Multiprocessing parameters 
    "nprocs" : [1,2,3,4], 
}

# define the loss function
loss_function = torch.nn.L1Loss()

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
        self.mask_A = torch.nn.parameter.Parameter( self.mask_A,requires_grad=False )
        self.mask_B = torch.nn.parameter.Parameter( self.mask_B,requires_grad=False )

    def forward(self,input):
        return (input*self.mask_A) * self.am_mul(input*self.mask_B) \
                                   + self.am_add(input*self.mask_B)

def benchmark(rank,data_size,dim,nprocs,train_time_per_epoch, infer_time_per_epoch):
    r"""
        data_size: int
            Number of training data points
        dim: int
            Dimension of the system
        train_time_per_epoch: torch.tensor
            Tensor to store the timing. Must be of size (rank,epoch)
        infer_time_per_epoch: torch.tensor
            Tensor to store the timing. Must be of size (rank,epoch)

        This Benchmark function mimics the training and inference (validation) of
        the Random Affine Coupling Layer define above. The training time per epoch
        and validation time in each epoch is measured and averaged.
        Intend to call this function for the different data_sizes and dimensions
        we want to include into the benchmark.

    """
    # get device 
    rank_device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device {rank_device}")

    # create default process group 
    torch.distributed.init_process_group("gloo",rank=rank,world_size=nprocs)
    # create local model 
    model = torch.nn.Sequential(*([RandomAffineCouplingLayer(dim)]*PARAMS["ncl"])).to(device=rank_device)
    # construct data distributed parallel model 
    ddp_model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[rank])
    # define the optimizer 
    optimizer = torch.optim.AdamW(ddp_model.parameters(),lr=PARAMS["lr"],weight_decay=PARAMS["wd"])
    # create data
    rdat = (torch.rand((data_size,dim),device = rank_device),torch.rand((data_size,dim),device = rank_device))
    # put the data into the torch interface
    train_dat = torch.utils.data.TensorDataset(*rdat)
    train_datLoader = torch.utils.data.DataLoader(train_dat,batch_size=data_size)

     # storage for the timing data
    bench_epoch_infer = torch.zeros(size=(PARAMS["epochs"],))
    bench_epoch_train = torch.zeros(size=(PARAMS["epochs"],))

    # Note: The epochs here are just a tool to generate statistics of the timing
    #       Correctness of the model is not tested in this benchmark
    for epoch in range(PARAMS["epochs"]):
        s_epoch = time_ns() * 1e-9
        # # there is only one minibatch
        for data,label in train_datLoader:
            # make a prediction
            pred = ddp_model(data)
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

    infer_time_per_epoch[rank] = bench_epoch_infer.mean()
    train_time_per_epoch[rank] = bench_epoch_train.mean()


if __name__ == "__main__":

    # create arrays to store the train/inference time
    timing_train = torch.zeros(size=( len(PARAMS['nprocs']),len(PARAMS['dim']),len(PARAMS['data size'])) )
    timing_infer = torch.zeros(size=( len(PARAMS['nprocs']),len(PARAMS['dim']),len(PARAMS['data size'])) )

   
    # create a figure to plot the benchmark results
    fig = plt.figure(figsize=(8.0,5.85)) #(width,height)
    train_ax = fig.add_subplot(1,2,1)
    infer_ax = fig.add_subplot(1,2,2)

    fig_strgScal = plt.figure(figsize=(8.0,5.85)) #(width,height)
    strgScal_train_ax = fig_strgScal.add_subplot(1,2,1)
    strgScal_infer_ax = fig_strgScal.add_subplot(1,2,2)

    fig_weakScal = plt.figure(figsize=(8.0,5.85)) #(width,height)
    weakScal_train_ax = fig_weakScal.add_subplot(1,2,1)
    weakScal_infer_ax = fig_weakScal.add_subplot(1,2,2)

    for i_np, nprocs in enumerate(PARAMS['nprocs']):
        train_time_per_epoch = torch.zeros(size=(nprocs,))
        infer_time_per_epoch = torch.zeros(size=(nprocs,))

        for i_d,dim in enumerate(PARAMS['dim']):
            for i_ds,data_size in enumerate(PARAMS['data size']): 
                print(f"Benchmarking dimension: {dim:g} with data size: {data_size:g} on {nprocs} gpus")
                # spawn the ddp training
                torch.multiprocessing.spawn(
                    benchmark,
                    args=(data_size,dim,nprocs,train_time_per_epoch,infer_time_per_epoch),
                    nprocs = nprocs,
                    join = True
                )
                # average the output over each process
                #print(f"train_time_per_epoch = {train_time_per_epoch}")
                #print(f"infer_time_per_epoch = {infer_time_per_epoch}")
                timing_train[i_np,i_d,i_ds] = train_time_per_epoch.mean(dim=0)
                timing_infer[i_np,i_d,i_ds] = infer_time_per_epoch.mean(dim=0)
            


    print(f"timing_train = {timing_train}")
    print(f"timing_infer = {timing_infer}")

    colors = ['b','g','r','c','m','y','k']
    line_types = ['-','--','-.',':']
    marker_types = ['.','o','^','s','p','*','x','+','d','1']

    # plot the timing results by data size (scaling)
    for i_d in range(len(PARAMS['dim'])):
        for i_np in range(len(PARAMS['nprocs'])):
            train_ax.plot(PARAMS["data size"],timing_train[i_np,i_d,:]*1e+3,'.',linestyle=line_types[i_np],c=colors[i_d],label=f"System dimension = {dim}, #GPUs = {PARAMS['nprocs'][i_np]}")
            infer_ax.plot(PARAMS["data size"],timing_infer[i_np,i_d,:]*1e+3,'.',linestyle=line_types[i_np],c=colors[i_d],label=f"System dimension = {dim}, #GPUs = {PARAMS['nprocs'][i_np]}")
        
    # plot the timing results by number of GPUs (weak scaling)
    for i_d in range(len(PARAMS['dim'])):
        for i_ds in range(len(PARAMS['data size'])):
            weakScal_train_ax.plot(PARAMS["nprocs"],timing_train[:,i_d,i_ds]*1e+3,marker_types[i_ds],c=colors[i_d],linestyle='-',label=f"System dimension = {dim}, data size = {PARAMS['data size'][i_ds]}")
            weakScal_infer_ax.plot(PARAMS["nprocs"],timing_infer[:,i_d,i_ds]*1e+3,marker_types[i_ds],c=colors[i_d],linestyle='-',label=f"System dimension = {dim}, data size = {PARAMS['data size'][i_ds]}")
        
    # plot the timing results by number of GPUs at fixed data size per Gpu (strong scaling)
    strgScal_train = torch.Tensor(size=(len(PARAMS["nprocs"]),))
    strgScal_infer = torch.Tensor(size=(len(PARAMS["nprocs"]),))
    for i_d in range(len(PARAMS['dim'])):
        for i_np in range(len(PARAMS['nprocs'])):
            strgScal_train[i_np] = timing_train[i_np,i_d,i_np]*1e+3 
            strgScal_infer[i_np] = timing_infer[i_np,i_d,i_np]*1e+3 
                
                
        strgScal_train_ax.plot(PARAMS["nprocs"], strgScal_train,'.-',c=colors[i_d],label=f"System dimension = {dim}, data size = 1e+4/GPU")
        strgScal_infer_ax.plot(PARAMS["nprocs"], strgScal_infer,'.-',c=colors[i_d],label=f"System dimension = {dim}, data size = 1e+4/GPU")


    # Post Process the plots
    infer_ax.set_title("DDP Scaling: Inference")
    train_ax.set_title("DDP Scaling: Training")
    
    weakScal_infer_ax.set_title("DDP Weak Scaling: Inference")
    weakScal_train_ax.set_title("DDP Weak Scaling: Training")
    
    strgScal_infer_ax.set_title("DDP Strong Scaling: Inference")
    strgScal_train_ax.set_title("DDP Strong Scaling: Training")       

    train_ax.set_xlabel("Training data size")
    infer_ax.set_xlabel("Training data size")

    weakScal_infer_ax.set_xlabel("Number GPUs")
    weakScal_train_ax.set_xlabel("Number GPUs")

    strgScal_infer_ax.set_xlabel("Number GPUs")    
    strgScal_train_ax.set_xlabel("Number GPUs")
    
    # only on the left plot
    train_ax.set_ylabel("Average Execution Time per Epoch [ms]")

    weakScal_train_ax.set_ylabel("Average Execution Time per Epoch [ms]")

    strgScal_train_ax.set_ylabel("Average Execution Time per Epoch [ms]")

    train_ax.grid()
    infer_ax.grid()

    weakScal_infer_ax.grid()
    weakScal_train_ax.grid()

    strgScal_infer_ax.grid()
    strgScal_train_ax.grid()

    handles, labels = train_ax.get_legend_handles_labels()
    strgScal_handles, strgScal_labels = strgScal_train_ax.get_legend_handles_labels()
    weakScal_handles, weakScal_labels = weakScal_train_ax.get_legend_handles_labels()

    fig.legend(handles,labels,bbox_to_anchor=(0.7, 0.25),
          ncol=1, fancybox=True, shadow=True)

    fig_strgScal.legend(strgScal_handles,strgScal_labels,bbox_to_anchor=(0.7, 0.25),
          ncol=1, fancybox=True, shadow=True)

    fig_weakScal.legend(weakScal_handles,weakScal_labels,bbox_to_anchor=(0.7, 0.25),
          ncol=1, fancybox=True, shadow=True) 


    fig.tight_layout(rect=[0,0.25,1,1])
    fig_strgScal.tight_layout(rect=[0,0.25,1,1])
    fig_weakScal.tight_layout(rect=[0,0.25,1,1])

    fn = "results/GPU_DDP"

    fn_train=fn+f"train_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.csv"
    fn_infer=fn+f"infer_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.csv"
    fn_scal =fn+f"Scaling_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.pdf"
    fn_strgScal =fn+f"StrongScaling_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.pdf"
    fn_weakScal =fn+f"WeakScaling_ncl{PARAMS['ncl']:g}_nll{PARAMS['nll']:g}_epochs{PARAMS['epochs']:g}.pdf"

    fig.savefig(fn_scal)
    fig_strgScal.savefig(fn_strgScal)
    fig_weakScal.savefig(fn_weakScal)

    # save the data in csv format using pandas as pd
    timing_train = pd.DataFrame(timing_train)
    timing_infer = pd.DataFrame(timing_infer)

    
    timing_train.to_csv(fn_train,sep=" ")
    timing_infer.to_csv(fn_infer,sep=" ")

    #np.savetxt(fn_train,timing_train)
    #np.savetxt(fn_infer,timing_infer)
    
