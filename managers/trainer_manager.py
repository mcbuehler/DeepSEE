from torch.nn.utils import clip_grad_value_

from managers.base_manager import BaseManager


class TrainerManager(BaseManager):
    """
    TrainerManager creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        super().__init__(opt, create_model=True)
        assert opt.isTrain
        self.optimizer_G, self.optimizer_D = \
                self.sr_model.create_optimizers(opt)
        self.old_lr = opt.lr

        self.generated = None

        # A dictionary with logging information to display to the user.
        self.logs = {}

    def get_logs(self):
        return {**self.logs, **self.sr_model_on_one_gpu.get_logs()}

    def preprocess_input(self, data):
        data = super().preprocess(data, from_dataloader=True)
        return data

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        data_preprocessed = self.preprocess_input(data)
        g_losses, generated = self.sr_model(data_preprocessed, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()

        if self.opt.gradient_clip > 0:
            clip_grad_value_(self.sr_model.parameters(),
                             self.opt.gradient_clip)

        self.optimizer_G.step()

        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        data_preprocessed = self.preprocess_input(data)
        d_losses = self.sr_model(data_preprocessed, mode='discriminator')

        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()

        if self.opt.gradient_clip > 0:
            clip_grad_value_(self.sr_model.parameters(),
                             self.opt.gradient_clip)

        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def save(self, epoch):
        self.sr_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
