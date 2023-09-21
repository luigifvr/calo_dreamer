# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
import os

# Other functions of project
from Util.util import *
from data_util import get_loaders
from transforms import *


class GenerativeModel(nn.Module):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted

    """
    def __init__(self, params, device):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.params = params
        self.device = device
        self.dim = self.params["dim"]
        self.conditional = get(self.params,'conditional',False)

        self.batch_size = self.params["batch_size"]
        self.batch_size_sample = get(self.params, "batch_size_sample", self.batch_size)

        self.epoch = get(self.params, "total_epochs", 0)
        self.net = self.build_net()
        self.iterations = get(self.params,"iterations", 1)
        self.regular_loss = []
        self.kl_loss = []

        self.runs = get(self.params, "runs", 0)
        self.iterate_periodically = get(self.params, "iterate_periodically", False)
        self.validate_every = get(self.params, "validate_every", 50)

        #init preprocessing
        self.transforms = get_transformations(params.get('transforms', None))
        self.train_loader, self.val_loader, self.bounds = get_loaders(params.get('hdf5_file'),
                                                                    params.get('particle_type'),
                                                                    params.get('xml_filename'),
                                                                    params.get('val_frac'),
                                                                    params.get('batch_size'),
                                                                    self.transforms,
                                                                    params.get('eps', 1.e-10),
                                                                    device=device,
                                                                    shuffle=True,
                                                                    width_noise=params.get('width_noise', 1.e-6))

    def build_net(self):
        pass

    def prepare_training(self):
        print("train_model: Preparing model training")
        self.use_scheduler = get(self.params, "use_scheduler", False)
        self.train_losses = np.array([])
        self.train_losses_epoch = np.array([])
        self.val_losses = np.array([])
        self.val_losses_epoch = np.array([])

        self.n_trainbatches = len(self.train_loader)
        self.n_traindata = self.n_trainbatches*self.batch_size
        self.set_optimizer(steps_per_epoch=self.n_trainbatches)

        self.sample_periodically = get(self.params, "sample_periodically", False)
        if self.sample_periodically:
            self.sample_every = get(self.params, "sample_every", 1)
            self.sample_every_n_samples = get(self.params, "sample_every_n_samples", 100000)
            print(f'train_model: sample_periodically set to True. Sampling {self.sample_every_n_samples} every'
                  f' {self.sample_every} epochs. This may significantly slow down training!')

        self.log = get(self.params, "log", True)
        if self.log:
            log_dir = os.path.join(self.params["out_dir"], "logs")
            self.logger = SummaryWriter(log_dir)
            print(f"train_model: Logging to log_dir {log_dir}")
        else:
            print("train_model: log set to False. No logs will be written")

    def set_optimizer(self, steps_per_epoch=1, params=None):
        """ Initialize optimizer and learning rate scheduling """
        if params is None:
            params = self.params

        self.optimizer = torch.optim.AdamW(
                self.net.parameters(),
                lr = params.get("lr", 0.0002),
                betas = params.get("betas", [0.9, 0.999]),
                eps = params.get("eps", 1e-6),
                weight_decay = params.get("weight_decay", 0.)
                )

        self.lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")
        if self.lr_sched_mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size = params["lr_decay_epochs"],
                    gamma = params["lr_decay_factor"],
                    )
        elif self.lr_sched_mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor = 0.8,
                    patience = 50,
                    cooldown = 100,
                    threshold = 5e-5,
                    threshold_mode = "rel",
                    verbose=True
                    )
        elif self.lr_sched_mode == "one_cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                params.get("max_lr", params["lr"]*10),
                epochs = params.get("cycle_epochs") or params["n_epochs"],
                steps_per_epoch=steps_per_epoch,
                )
        elif self.lr_sched_mode == "cycle_lr":
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr = params.get("lr", 1.0e-4),
                max_lr = params.get("max_lr", params["lr"]*10),
                step_size_up= params.get("step_size_up", 2000),
                mode = params.get("cycle_mode", "triangular"),
                cycle_momentum = False,
                    )
        elif self.lr_sched_mode == "multi_step_lr":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[2730, 8190, 13650, 27300],
                    gamma=0.5
                    )

    def run_training(self):

        self.prepare_training()
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        for e in range(n_epochs):
            t0 = time.time()

            self.epoch = past_epochs + e
            self.train()
            self.train_one_epoch()

            if (self.epoch + 1) % self.validate_every == 0:
                self.eval()
                self.validate_one_epoch()

            if self.sample_periodically:
                if (self.epoch + 1) % self.sample_every == 0:
                    self.eval()

                    # if true then i * bayesian samples will be drawn, else just 1
                    iterations = self.iterations if self.iterate_periodically else 1
                    bay_samples = []
                    for i in range(0, iterations):
                        sample, c = self.sample_n(self.sample_every_n_samples)
                        bay_samples.append(sample)

                    samples = np.concatenate(bay_samples)
                    self.plot_samples(samples=samples, conditions=c)

            # save model periodically, useful when trying to understand how weights are learned over iterations
            if get(self.params,"save_periodically",False):
                if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
                    torch.save(self.state_dict(), f"models/model_epoch_{e+1}.pt")

            # estimate training time
            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")

    def train_one_epoch(self):
        # create list to save train_loss
        train_losses = np.array([])

        # iterate batch wise over input
        for batch_id, x in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            # calculate batch loss
            loss = self.batch_loss(x)

            if np.isfinite(loss.item()): # and (abs(loss.item() - loss_m) / loss_s < 5 or len(self.train_losses_epoch) == 0):
                loss.backward()
                self.optimizer.step()
                train_losses = np.append(train_losses, loss.item())
                if self.log:
                    self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)

                if self.use_scheduler:
                    self.scheduler.step()
                    if self.log:
                        self.logger.add_scalar("learning_rate", self.scheduler.get_last_lr()[0],
                                               self.epoch * self.n_trainbatches + batch_id)

            else:
                print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")

        self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
        self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
        if self.log:
            self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)
            if self.use_scheduler:
                self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0],
                                       self.epoch)

    def validate_one_epoch(self):
        val_losses = np.array([])

        # iterate batch wise over input
        with torch.no_grad():
            for batch_id, x in enumerate(self.val_loader):

                # calculate batch loss
                loss = self.batch_loss(x)

                val_losses = np.append(val_losses, loss.item())
                if self.log:
                    self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

            self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
            self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
            if self.log:
                self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)

    def batch_loss(self, x):
        pass

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1
                        sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
                        and 5, 3, 2, 1 times sample multiplier for the highest energies

        """
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [*ret, *np.tile(2. ** 19, 5), *np.tile(2. ** 20, 3), *np.tile(2. ** 21, 2), *np.tile(2. ** 22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret

    def sample_n(self):
        sample = []
        condition = self.generate_Einc_ds1().to(self.device)
        batch_size_sample = get(self.params, "batch_size_sample", 10000)
        condition_loader = DataLoader(dataset=condition, batch_size=batch_size_sample, shuffle=False)

        for _, batch in enumerate(condition_loader):
            sample.append(self.sample_batch(batch).detach().cpu().numpy())
        return np.concatenate(sample), condition.detach().cpu().numpy()

    def sample_batch(self, batch):
        pass

    def plot_samples(self, samples, conditions, finished=False):
        transforms = self.transforms

        for fn in transforms[::-1]:
            samples, conditions = fn(samples, conditions, rev=True)


