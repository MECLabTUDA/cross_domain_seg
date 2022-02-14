from mp.agents.agent import Agent
import random
import math
import statistics as stat
import os
from mp.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state
from mp.utils.load_restore import pkl_dump, pkl_load
import torchio

class SA_Agent(Agent):

    def get_inputs_targets(self, data, getName=False):
        r"""Prepares a data batch.

        Args:
            data (tuple): a dataloader item, possibly in cpu

        Returns (tuple): preprocessed data in the selected device.
        """
        inputs, targets, _, _, name, x_affine = data #x, y, pose, identity, name, x_affine
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        inputs = self.model.preprocess_input(inputs)       

        if not getName:
            return inputs, targets.float()
        else:
            return inputs, targets.float(), name, x_affine

    def style_transfer(self, results, train_dataloader_s, train_dataloader_t,
        eval_dataloader_s=None, eval_dataloader_t=None,
        save_path=None):

        self.model.eval()

        save_dest = os.path.join(save_path, 'TestImages', 'imagesTr')
        print(save_dest)
        for i, data in enumerate(zip(train_dataloader_s, train_dataloader_t)):

            inputs_s, targets_s, name_s, affine_s = self.get_inputs_targets(data[0], getName = True)
            inputs_t, targets_t, name_t, affine_t = self.get_inputs_targets(data[1], getName = True)
            self.model.forward(inputs_s, inputs_t)

            img_trans = self.model.ts 
            img_trans = img_trans.permute((0, 2, 3, 1))
            img_rec = img_trans.data.detach().cpu().numpy() 
            self.save_synthetic_image(save_dest, str(name_t[0]), img_rec, affine_t)

        for i, data in enumerate(zip(eval_dataloader_s, eval_dataloader_t)):

            inputs_s, targets_s, name_s, affine_s = self.get_inputs_targets(data[0], getName = True)
            inputs_t, targets_t, name_t, affine_t = self.get_inputs_targets(data[1], getName = True)
            self.model.forward(inputs_s, inputs_t)

            img_trans = self.model.ts 
            img_trans = img_trans.permute((0, 2, 3, 1))
            img_rec = img_trans.data.detach().cpu().numpy() 
            self.save_synthetic_image(save_dest, str(name_t[0]), img_rec, affine_t)

        return 

    def save_synthetic_image(self, save_dest, name, img_rec, affine):
        os.makedirs(save_dest, exist_ok=True)
        img = torchio.Image(tensor=img_rec, affine=affine.squeeze())
        path = os.path.join(save_dest, name + ".nii.gz")
        img.save(path)

    # Allows for saving a list of optimizers if necessary
    def save_state(self, states_path, state_name, optimizers=None, optimizer_names=None, overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and 
        overwrite=False.
        """
        if states_path is not None:
            state_full_path = os.path.join(states_path, state_name)
            if os.path.exists(state_full_path):
                if not overwrite:
                    raise FileExistsError
                shutil.rmtree(state_full_path)
            os.makedirs(state_full_path)
            save_model_state(self.model, 'model', state_full_path)
            pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
            for idx, optimizer in enumerate(optimizers):
                save_optimizer_state(optimizer, 'optimizer_' + optimizer_names[idx], state_full_path)

    def restore_state(self, states_path, state_name, optimizer_G=None, optimizer_D=None):
        r"""Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful.
        """
        state_full_path = os.path.join(states_path, state_name)
        try:
            correct_load = load_model_state(self.model, 'model', state_full_path, device=self.device)
            assert correct_load
            print(state_full_path)
            agent_state_dict = pkl_load('agent_state_dict', state_full_path)
            assert agent_state_dict is not None
            self.agent_state_dict = agent_state_dict
            if optimizer_G is not None: 
                load_optimizer_state(optimizer_G, 'optimizer_G', state_full_path, device=self.device)
            if optimizer_D is not None: 
                load_optimizer_state(optimizer_D, 'optimizer_D', state_full_path, device=self.device)
            if self.verbose:
                print('State {} was restored'.format(state_name))
            return True
        except:
            print('State {} could not be restored'.format(state_name))
            return False

    def test(self, results, epoch, eval_dataloader_s, eval_dataloader_t, save_path, nr_samples=1):
        samples = random.sample(range(len(eval_dataloader_s)), nr_samples)
        loss_G = []
        loss_D = []
        for i, data in enumerate(zip(eval_dataloader_s, eval_dataloader_t)):

            inputs_s, targets_s = self.get_inputs_targets(data[0])
            inputs_t, targets_t = self.get_inputs_targets(data[1])
            self.model.forward(inputs_s, inputs_t)
            i_loss_G = self.model.optimize_DG_parameters(update_Weights=False)  
            loss_G.append(i_loss_G)
            
            if i in samples:
                self.model.save_result(save_path, epoch + 1, 'test')

        results.add(epoch + 1, 'Loss_G_Mean', stat.mean(loss_G), 'test')

    def train(self, results, optimizer_G, optimizer_D, loss_f, train_dataloader_s, train_dataloader_t,
        init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
        eval_dataloader_s=None, eval_dataloader_t=None, eval_interval=10, 
        save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """


        for epoch in range(init_epoch, init_epoch+nr_epochs):

            print("EPOCH: " + str(epoch))
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            nr_samples = 1
            samples = random.sample(range(len(train_dataloader_s)), nr_samples)

            loss_G = []
            loss_D = []

            for i, data in enumerate(zip(train_dataloader_s, train_dataloader_t)):#for i, data in enumerate(train_dataloader_s):

                inputs_s, targets_s = self.get_inputs_targets(data[0])
                inputs_t, targets_t = self.get_inputs_targets(data[1])

                self.model.forward(inputs_s, inputs_t)
                i_loss_G = self.model.optimize_DG_parameters()
                loss_G.append(i_loss_G)

                if i in samples:
                    self.model.save_result(save_path, epoch + 1, 'train')

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                results.add(epoch + 1, 'Loss_G_Mean', stat.mean(loss_G), 'train')
                self.test(results, epoch, eval_dataloader_s, eval_dataloader_t, save_path)

            # Save agent and optimizer state
            if ((epoch + 1) % save_interval == 0 or init_epoch+nr_epochs == epoch) and save_path is not None:
                print("SaveAgent")
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), [optimizer_G, optimizer_D], ['G', 'D'])