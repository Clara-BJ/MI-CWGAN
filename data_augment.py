import numpy as np
from GANs import Multi_Generator, Multi_Discriminator
from deep_learning import Multi_Gan_Trainer
import torch
import gc

in_feature = 12
augment_times = 6

def multi_data_augment(X, Y):
    input_data = [X[0], np.concatenate([X[1], X[2]], axis=1)]
    eeg_X_list = [input_data[0]]
    fnirs_X_list = [input_data[1]]
    new_Y_list = [Y]

    n_classes = len(np.unique(Y))
    n_electrodes_eeg = X[0].shape[1]
    n_electrodes_fnirs = X[1].shape[1]

    in_feature = 12

    multi_G = Multi_Generator(n_electrodes_eeg=n_electrodes_eeg, n_electrodes_fnirs=n_electrodes_fnirs,
                               n_classes=n_classes, in_features_length=in_feature, )
    multi_D = Multi_Discriminator(n_electrodes_eeg=n_electrodes_eeg, n_electrodes_fnirs=n_electrodes_fnirs,
                                   n_classes=n_classes, )

    Multi_Gan_Trainer.mean = 0
    Multi_Gan_Trainer.std = 0.02
    trainer = Multi_Gan_Trainer(multi_G, multi_D)

    model_G,model_D = trainer.train(multi_G, multi_D, X, Y, n_classes)
    for augment_index in range(augment_times):

        for c in range(n_classes):
            Y_c = Y[Y == c]
            c_index = np.where(Y == c)
            X_eeg_c = input_data[0][c_index]
            X_fnirs_c = input_data[1][c_index]
            n_trials_c = Y_c.shape[0]
            test_z = torch.randn(n_trials_c, 300)
            test_y_label = torch.zeros(n_trials_c, n_classes)
            test_y_label[:, c] = 1
            model_G.eval()

            X_new_c = model_G(test_z, test_y_label)
            X_new_c_arr1 = torch.squeeze(X_new_c[0]).cpu().detach().numpy() + X_eeg_c
            fnirs = (X_new_c[1] - torch.mean(X_new_c[1])) / torch.std(X_new_c[1])
            fnirs = fnirs / 500
            X_new_c_arr2 = torch.squeeze(fnirs).cpu().detach().numpy() + X_fnirs_c

            eeg_X_list.append(X_new_c_arr1)
            fnirs_X_list.append(X_new_c_arr2)
            new_Y_list.append(Y_c)
        torch.cuda.empty_cache()
    eeg_X = np.concatenate(eeg_X_list)
    fnirs_X = np.concatenate(fnirs_X_list)
    new_Y = np.concatenate(new_Y_list)

    gc.collect()
    return [eeg_X, fnirs_X[:, :24, :], fnirs_X[:, 24:, :]], new_Y