#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()

                predictions = self.model(batch_input, mask)
                batch_input_arr = batch_input.cpu().detach().numpy()
                batch_target_arr = batch_target.cpu().detach().numpy()
                predictions_arr = predictions.cpu().detach().numpy()
                mask_arr = mask.cpu().detach().numpy()
                loss = 0
                for p in predictions:
                    a = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                predictions_arr = predictions.cpu().detach().numpy()
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


class MultiStageModel_conf(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel_conf, self).__init__()
        self.stage1 = SingleStageModel_conf(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel_conf(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(out * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel_conf(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel_conf, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class Trainer_conf:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel_conf(num_blocks, num_layers, num_f_maps*2, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)
                batch_target_arr = batch_target.cpu().detach().numpy()
                predictions_arr = predictions.cpu().detach().numpy()
                mask_arr = mask.cpu().detach().numpy()
                loss = 0
                for p in predictions:
                    loss += self.mse(p.to(torch.float32), batch_target.to(torch.float32))
                    # loss += 0.15*torch.mean(torch.clamp(self.mse(p[:, :, 1:], F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                predictions_arr = np.array(predictions.cpu())
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = results_dir + "/" + f_name + '.npy'
                np.save(f_ptr, predictions_arr)
                _, predicted = torch.max(predictions[-1].data, 1)
                # predicted = predicted.squeeze()
                # recognition = []
                # for i in range(len(predicted)):
                #     recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                # f_ptr.write("### Frame level recognition: ###\n")
                # f_ptr.write(' '.join(recognition))
                # f_ptr.close()


class Trainer_conf_ED:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        # self.model = MultiStageModel_conf(num_blocks, num_layers, num_f_maps*2, dim, num_classes)
        self.model = EDTCN(dim, num_classes)

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                loss = 0
                for p in predictions:
                    loss += self.mse(p.to(torch.float32), batch_target.to(torch.float32))
                    # loss += 0.15*torch.mean(torch.clamp(self.mse(p[:, :, 1:], F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                batch_target_arr = batch_target.cpu().numpy()

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions.data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                predictions_arr = np.array(predictions.cpu())
                _, predicted = torch.max(predictions[-1].data, 1)
                # predicted = predicted.squeeze()
                # recognition = []
                # for i in range(len(predicted)):
                #     recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                # f_name = vid.split('/')[-1].split('.')[0]
                # f_ptr = open(results_dir + "/" + f_name, "w")
                # f_ptr.write("### Frame level recognition: ###\n")
                # f_ptr.write(' '.join(recognition))
                # f_ptr.close()


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(
        self,
        in_channel: int,
        n_classes: int,
        kernel_size: int = 25,
        mid_channels = [128, 160],
        # **kwargs: Any
    ) -> None:
        """
        Args:
            in_channel: int. the number of the channels of input feature
            n_classes: int. output classes
            kernel_size: int. 25 is proposed in the original paper
            mid_channels: list. the list of the number of the channels of the middle layer.
                        [96 + 32*1, 96 + 32*2] is proposed in the original paper
        Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channel,
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1],
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2,), mode="nearest")
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1,), mode="nearest")
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x
