# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os, time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from matplotlib.backends.backend_pdf import PdfPages
import os
import sys

# Other functions of project
from Util.util import *
from data_util import get_loaders
from datasets import CaloChallengeDataset
from plotting_util import *
from transforms import *
from challenge_files import *
from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined

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
    def __init__(self, params, device, doc):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.doc = doc
        self.params = params
        self.device = device
        self.shape = self.params['shape']#get(self.params,'shape')
        self.conditional = get(self.params,'conditional',False)
        self.single_energy = get(self.params, 'single_energy', None) # Train on a single energy

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
                                                                    width_noise=params.get('width_noise', 1.e-6),
                                                                    single_energy=params.get('single_energy', None))

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
            log_dir = self.doc.basedir
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
                pct_start=params.get("cycle_pct_start", 0.3)
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
        elif self.lr_sched_mode == "CosineAnnealing":
            n_epochs = params.get("cycle_epochs") or params["n_epochs"]
            eta_min = params.get( "eta_min", 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=n_epochs * steps_per_epoch,
                eta_min=eta_min
            )

    def run_training(self):

        self.prepare_training()
        samples = []
        n_epochs = get(self.params, "n_epochs", 100)
        past_epochs = get(self.params, "total_epochs", 0)
        print(f"train_model: Model has been trained for {past_epochs} epochs before.")
        print(f"train_model: Beginning training. n_epochs set to {n_epochs}")
        t_0 = time.time()
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
                        sample, c = self.sample_n()
                        bay_samples.append(sample)

                    samples = np.concatenate(bay_samples)
                    self.plot_samples(samples=samples, conditions=c, name=self.epoch, energy=self.params['single_energy'])

            # save model periodically, useful when trying to understand how weights are learned over iterations
            if get(self.params,"save_periodically",False):
                if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
                    self.save(epoch=f"self.epoch")

            # estimate training time
            if e==0:
                t1 = time.time()
                dtEst= (t1-t0) * n_epochs
                print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h")
            sys.stdout.flush()
        t_1 = time.time()
        traintime = t_1 - t_0
        self.params['traintime'] = traintime
        print(
            f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.", flush=True)
        
        #save final model
        print("train_model: Saving final model: ", flush=True)
        self.save()
        # generate and plot samples at the end
        if get(self.params, "sample", True):
            print("generate_samples: Start generating samples", flush=True)
            t_0 = time.time()
            samples, c = self.sample_n()
            t_1 = time.time()
            sampling_time = t_1 - t_0
            self.params["sampling_time"] = sampling_time
            print(f"generate_samples: Finished generating {len(samples)} samples after {sampling_time} s.", flush=True)
            self.plot_samples(samples=samples, conditions=c, energy=self.params['single_energy'])

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
                # if self.log:
                #     self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)

                if self.use_scheduler:
                    self.scheduler.step()
                    # if self.log:
                    #     self.logger.add_scalar("learning_rate", self.scheduler.get_last_lr()[0],
                    #                            self.epoch * self.n_trainbatches + batch_id)

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
                # if self.log:
                #     self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

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
        self.eval()

        if self.net.bayesian:
            self.net.map = get(self.params, "fix_mu", False)
            for bay_layer in self.net.bayesian_layers:
                bay_layer.random = None
        sample = []

        Einc = torch.tensor(
            # Ayo: TODO: Generalise number of samples to generate
            # Ayo: TODO: Handle single energy option for datasets 2 & 3
            10**np.random.uniform(3, 6, size=10**5) 
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy = torch.empty(1, *self.params['shape'])
        transformed_cond = torch.clone(Einc)
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batch_size_sample = get(self.params, "batch_size_sample", 10000)
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
        )
        
        if self.params['model_type'] == 'shape': # sample u_i's if self is a shape model
            # load energy model
            energy_model = self.load_other(self.params['energy_model'])
            energy_model.eval()
            if self.params.get('sample_us', True):
                # sample us
                u_samples = torch.from_numpy(np.vstack([ # Ayo: TODO: avoid cast to numpy (it happens elsewhere too)
                    energy_model.sample_batch(c) for c in transformed_cond_loader
                ])).to(self.device)
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:
                transformed_cond = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=self.transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
                ).energy
            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batch_size_sample, shuffle=False
            )
                
        sample = np.vstack([self.sample_batch(c) for c in transformed_cond_loader])

        return sample, transformed_cond.detach().cpu()

    def sample_batch(self, batch):
        pass

    def plot_samples(self, samples, conditions, name="", energy=None):
        transforms = self.transforms
        samples = torch.from_numpy(samples) # since transforms expect torch.tensor

        if self.params['model_type'] == 'energy':
            reference = CaloChallengeDataset(
                self.params.get('eval_hdf5_file'),
                self.params.get('particle_type'),
                self.params.get('xml_filename'),
                transform=transforms, # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
                single_energy=self.single_energy
            ).layers
            
            # postprocess
            for fn in transforms[::-1]:
                if fn.__class__.__name__ != 'NormalizeByElayer':
                    samples, _ = fn(samples, conditions, rev=True)
                    reference, _ = fn(reference, conditions, rev=True)
            # clip u_i's (except u_0) to [0,1] 
            samples[:,1:] = torch.clip(samples[:,1:], min=0., max=1.)
            reference[:,1:] = torch.clip(reference[:,1:], min=0., max=1.)
            
            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=self.doc
            )
        else:
            # postprocess
            for fn in transforms[::-1]:
                samples, conditions = fn(samples, conditions, rev=True)
            self.save_sample(samples, conditions, name=name)
            script_args = (
                f"-i {self.doc.basedir}/samples{name}.hdf5 "
                f"-r {self.params['eval_hdf5_file']} -m all --cut 15.15e-3 "
                f"-d {self.params['eval_dataset']} --output_dir {self.doc.basedir}/final/"
            ) + (f" --energy {energy}" if energy is not None else '')
            evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name=""):
        """Save sample in the correct format"""
    
        save_file = h5py.File(self.doc.get_file(f'samples{name}.hdf5'), 'w')
        save_file.create_dataset('incident_energies', data=energies)
        save_file.create_dataset('showers', data=sample)
        save_file.close()            
 
    def save(self, epoch=""):
        """ Save the model, and more if needed"""
        torch.save({#"opt": self.optim.state_dict(),
                    "net": self.net.state_dict(),
                    #"losses": self.losses_test,
                    #"learning_rates": self.learning_rates,
                    }#"epoch": self.epoch}
                    , self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """ Load the model, and more if needed"""
        name = self.doc.get_file(f"model{epoch}.pt")
        state_dicts = torch.load(name, map_location=self.device)
        self.net.load_state_dict(state_dicts["net"])

        #self.losses_test = state_dicts.get("losses", {})
        #self.learning_rates = state_dicts.get("learning_rates", [])
        #self.epoch = state_dicts.get("epoch", 0)
        #self.optim.load_state_dict(state_dicts["opt"])
        self.net.to(self.device)

    def load_other(self, model_dir):
        """ Load a different model (e.g. to sample u_i's)"""

        with open(os.path.join(model_dir, 'params.yaml')) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        other = self.__class__(params, self.device, None) # or maybe self.__class__.__init__?
        # other.net = other.build_net()
        state_dicts = torch.load(os.path.join(model_dir, 'model.pt'), map_location=self.device)
        other.net.load_state_dict(state_dicts["net"])
        
        return other
