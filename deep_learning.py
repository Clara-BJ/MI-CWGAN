import torch
import numpy as np
from dataloader import MultiDataLoader, MultiModalDataset
import time
from torch.autograd import grad
import torch.nn.functional as F

class Gan_Trainer:
    def __init__(self, model_G, model_D, model_name, data_augment_params):
        self.optimizer_G = torch.optim.Adam(model_G.parameters(), lr=self.gan_params["lr_G"])
        self.optimizer_D = torch.optim.Adam(model_D.parameters(), lr=self.gan_params["lr_D"])
        self.data_augment_params = data_augment_params

    def train(self, model_G, model_D, X_train, Y_train, n_classes, save_path):
        n_epochs = 200
        batch_size = 32

        eeg_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train[0]), torch.from_numpy(Y_train))
        fnirs_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train[1]), torch.from_numpy(Y_train))
        multildataset = MultiModalDataset([eeg_data, fnirs_data])
        train_dataloader = MultiDataLoader(multildataset, batch_size=batch_size, shuffle=True,
                                           num_workers=1)

        train_hist = {}
        train_hist['D_losses'] = []
        train_hist['G_losses'] = []
        train_hist['per_epoch_ptimes'] = []
        train_hist['total_ptime'] = []

        print('training start!')
        i = 0
        start_time = time.time()
        save_epoch = self.gan_params["save_epoch"]

        for epoch in range(n_epochs):
            D_losses = []
            G_losses = []

            epoch_start_time = time.time()
            for [eeg_data, fnirs_data, y_] in train_dataloader:#按照batch拿数据
                eeg_data = eeg_data.to(self.device, dtype=torch.float)  # 将张量送入GPU并确定类型
                fnirs_data = fnirs_data.to(self.device, dtype=torch.float)  # 将张量送入GPU并确定类型
                y_ = F.one_hot(y_.to(torch.int64), num_classes=n_classes)
                y_ = y_.to(self.device, dtype=torch.float)  # 将张量送入GPU并确定类型
                x_ = [eeg_data, fnirs_data]

                i += 1
                model_D.zero_grad()
                mini_batch = x_[0].size()[0]

                D_result = self.forward_D(model_D, x_, label=y_)#判别器的前向传播
                D_real_loss = self.compute_loss_D(D_result, -1)

                # fake sample
                gen_signal, noise = self.generate_fake_data(model_G, mini_batch, label=y_)

                if self.gan_params["random_add"]:
                    random = 0.2 + 0.6 * np.random.random()
                    gen_signal = [random * gen_signal[0] + (1-random) * x_[0],
                                  random * gen_signal[1] + (1-random) * x_[1]]
                else :
                    fnirs = (gen_signal[1]-torch.mean(gen_signal[1]))/torch.std(gen_signal[1])

                    gen_signal = [gen_signal[0] + x_[0], fnirs + x_[1]]

                D_result = self.forward_D(model_D, gen_signal, label=y_)
                D_fake_loss = self.compute_loss_D(D_result, 1)
                gradient_penalty = self.gradient_penalty(mini_batch, x_, y_, gen_signal, model_D)

                self.backward_D(D_real_loss, D_fake_loss, gradient_penalty, D_losses)

                self.optimizer_D.step()

                if (i % self.gan_params["n_critic"]) == 0:
                    model_G.zero_grad()

                    y_label = self.make_label(mini_batch, n_classes)

                    G_result = self.generate_fake_data(model_G, mini_batch, noise=noise, label=y_label)
                    if self.gan_params["random_add"]:
                        random = 0.5 + 0.5 * np.random.random()
                        G_result = [random * G_result[0] + (1 - random) * x_[0],
                                    random * G_result[1] + (1 - random) * x_[1]]
                    else :
                        fnirs = (G_result[1] - torch.mean(G_result[1])) / torch.std(G_result[1])

                        G_result = [G_result[0] + x_[0], fnirs + x_[1]]
                    D_result = self.forward_D(model_D, G_result, label=y_label)
                    G_train_loss = self.compute_loss_D(D_result, -1)

                    self.backward_G(G_train_loss, G_losses)
                    torch.nn.utils.clip_grad_norm_(model_G.parameters(), max_norm=1.0)
                    self.optimizer_G.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (
                (epoch + 1), n_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                torch.mean(torch.FloatTensor(G_losses))))
            # show_result(model_G, (epoch + 1), test_z_, test_y_label_, n_times, show=True)
            train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
            train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
            if (epoch + 1) in save_epoch:
                path_G = save_path + "_" + str(epoch+1) + "_generator_param.pth"
                path_D = save_path + "_" + str(epoch+1) + "_discriminator_param.pth"
                torch.save(model_G.state_dict(), path_G)
                torch.save(model_D.state_dict(), path_D)
        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
            torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), n_epochs, total_ptime))
        print("Training finish!... save training results")
        path_G = save_path + "_generator_param.pth"
        path_D = save_path + "_discriminator_param.pth"
        torch.save(model_G.state_dict(), path_G)
        torch.save(model_D.state_dict(), path_D)
        return model_G, model_D

    def gradient_penalty(self, mini_batch, x_, y_ ,gen_signal, model_D):
        alpha = torch.rand((mini_batch, 1, 1, 1)).cuda()
        x_hat = alpha * x_ + (1 - alpha) * gen_signal
        pre_hat = model_D(x_hat, y_)

        gradients = grad(outputs=pre_hat, inputs=x_hat, grad_outputs=torch.ones(pre_hat.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = self.gan_params["gp"] * (
                    (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        return gradient_penalty

    def forward_D(self, model_D, x_, **kwargs):
        D_result = model_D(x_)
        return D_result

    def make_label(self, batch_size, n_classes):
        return None

    def generate_fake_data(self, model_G, batch_size=100, **kwargs):
        if "noise" not in kwargs:
            noise = np.random.normal(0, 1, (batch_size, self.gan_params["noise_dim"]))
            gen_signal = model_G(noise)
            return gen_signal, noise
        else:
            gen_signal = model_G(kwargs["noise"])
            return gen_signal

    def compute_loss_D(self, result, label):
        return label * torch.mean(result.squeeze())

    def backward_D(self, D_real_loss, D_fake_loss, gradient_penalty, losses):
        loss = D_real_loss + D_fake_loss + gradient_penalty
        loss.backward()
        losses.append(loss.item())

    def backward_G(self, G_train_loss, losses):
        G_train_loss.backward()
        losses.append(G_train_loss.item())

    def compute_gradient_penalty(self):
        return

class Multi_Gan_Trainer(Gan_Trainer):

    def make_label(self, batch_size, n_classes):
        y_ = (torch.rand(batch_size, 1) * n_classes).type(torch.LongTensor).squeeze()
        y_ = F.one_hot(y_, num_classes=n_classes).to(self.device, dtype=torch.float)
        return y_

    def forward_D(self, model_D, x_, **kwargs):
        D_result = model_D(x_, kwargs["label"])
        return D_result

    def backward_D(self, D_real_loss, D_fake_loss, gradient_penalty, losses):
        loss2 = D_real_loss[1] + D_fake_loss[1] + gradient_penalty[1]
        loss = loss2
        loss.backward()
        losses.append(loss.item())

    def compute_loss_D(self, result, label):
        r1 = super().compute_loss_D(result[0], label)
        r2 = super().compute_loss_D(result[1], label)
        r3 = super().compute_loss_D(result[2], label)
        return [r1, r2, r3]

    def backward_G(self, G_train_loss, losses):
        loss = G_train_loss[1]
        loss.backward()
        losses.append(loss.item())

    def generate_fake_data(self, model_G, batch_size=100, **kwargs):
        noise = torch.Tensor(np.random.normal(self.mean, self.std, (batch_size, self.gan_params["noise_dim"])))
        noise = noise.to(self.device, dtype=torch.float)

        gen_signal = model_G(noise, kwargs["label"])
        return gen_signal, noise

    def gradient_penalty(self, mini_batch, x_, y_, gen_signal, model_D):
        # gradient penalty
        alpha1 = torch.rand((mini_batch, 1, 1)).cuda(self.device)
        alpha2 = torch.rand((mini_batch, 1, 1)).cuda(self.device)

        x_hat1 = alpha1 * x_[0] + (1 - alpha1) * gen_signal[0]
        x_hat2 = alpha2 * x_[1] + (1 - alpha2) * gen_signal[1]
        pre_hat = model_D([x_hat1, x_hat2], y_)

        gradients = grad(outputs=pre_hat[0], inputs=x_hat1, grad_outputs=torch.ones(pre_hat[0].size()).cuda(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        g1 = self.gan_params["gp"] * (
                (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        gradients = grad(outputs=pre_hat[1], inputs=x_hat2, grad_outputs=torch.ones(pre_hat[1].size()).cuda(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        g2 = self.gan_params["gp"] * (
                (gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        return [g1, g2]

