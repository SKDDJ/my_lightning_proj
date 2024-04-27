# Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math

# import pdb
import random

import torch
import torch.nn.functional as F
from torch import nn


# import pysnooper
# 这段代码实现了一个基于卷积编码器-解码器架构的神经网络模型,用于对一维序列数据(如时间序列)进行编码和重构。
# one dimension convolutional encoder
class ODEncoder(nn.Module):
    # 定义了一个名为 ODEncoder 的卷积编码器类。在__init__方法中，初始化了一些参数，
    # 如输入维度列表 in_dim_list、下采样率 fold_rate、卷积核大小 kernel_size 和
    # 编码器各层的通道数 channel_list。然后，根据层数构建了一系列编码层，
    # 每一层由 build_layer 方法构造，并存储在 nn.ModuleList 中。
    def __init__(self, in_dim_list, fold_rate, kernel_size, channel_list):
        super().__init__()
        self.in_dim_list = in_dim_list
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size

        # insert the first layer
        channel_list = [channel_list[0]] + channel_list
        self.channel_list = channel_list
        encoder = nn.ModuleList()
        layer_num = len(channel_list) - 1  # default fixed layer_num

        for i in range(layer_num):
            if_last = False
            if_start = i == 0
            if i == layer_num - 1:
                if_last = True
            layer = self.build_layer(
                in_dim_list[i],
                kernel_size,
                fold_rate,
                channel_list[i],
                channel_list[i + 1],
                if_last,
                if_start,
            )
            encoder.append(layer)
        self.encoder = encoder

    # build_layer 方法构建单个编码层，包括 LeakyReLU 激活函数、InstanceNorm1d 归一化层、
    # Conv1d 卷积层等操作。如果是编码器的第一层，则使用 nn.Identity() 代替 LeakyReLU 激活函数。
    # 如果是最后一层，则在最后使用 Tanh 激活函数。
    def build_layer(
        self,
        in_dim,
        kernel_size,
        fold_rate,
        input_channel,
        output_channel,
        last=False,
        if_start=False,
    ):
        # first: if is the first layer of encoder
        layer = nn.Sequential(
            nn.LeakyReLU() if not if_start else nn.Identity(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(
                input_channel if not if_start else 1,
                input_channel,
                kernel_size,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(input_channel, output_channel, kernel_size, stride=fold_rate, padding=0),
            nn.Tanh() if last else nn.Identity(),
        )
        return layer

    # forward 方法定义了编码器的前向传播过程，将输入 x 依次通过所有编码层。
    def forward(self, x, **kwargs):
        for _, module in enumerate(self.encoder):
            x = module(x)
        # print("encoder")
        # print(x.shape)
        return x


class ODDecoder(nn.Module):
    # 定义了一个名为 ODDecoder 的卷积解码器类。结构与 ODEncoder 类似，
    # 不过通道数列表 channel_list 在最后追加了一个 1, 表示输出通道数为 1 (对应一维序列数据)。
    def __init__(self, in_dim_list, fold_rate, kernel_size, channel_list):
        super().__init__()
        self.in_dim_list = in_dim_list
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size

        # insert the first layer
        channel_list = channel_list + [1]
        self.channel_list = channel_list
        decoder = nn.ModuleList()
        layer_num = len(channel_list) - 1

        for i in range(layer_num):
            if_last = False
            if i == layer_num - 1:
                if_last = True
            layer = self.build_layer(
                in_dim_list[i],
                kernel_size,
                fold_rate,
                channel_list[i],
                channel_list[i + 1],
                if_last,
            )
            decoder.append(layer)
        self.decoder = decoder
        self.last_conv = nn.Conv1d(
            in_channels=1,  # 输入通道数
            out_channels=1,  # 输出通道数
            kernel_size=34,  # 核大小
            stride=1,  # 步长
            padding=0,
        )  # 填充

    # build_layer 方法构建单个解码层，包括 LeakyReLU 激活函数、InstanceNorm1d 归一化层、
    # ConvTranspose1d 上采样层和 Conv1d 卷积层等操作。
    def build_layer(self, in_dim, kernel_size, fold_rate, input_channel, output_channel, last):
        layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.ConvTranspose1d(
                input_channel, input_channel, kernel_size, stride=fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(
                input_channel,
                output_channel,
                kernel_size,
                stride=1,
                # padding=fold_rate if last else fold_rate - 1,
                padding=8,
            ),
        )

        return layer

    # forward 方法定义了解码器的前向传播过程，将输入 x 依次通过所有解码层。
    def forward(self, x, **kwargs):
        for i, module in enumerate(self.decoder):
            x = module(x)
        x = self.last_conv(x)
        # print("decode")
        # print(x.shape)
        # exit()
        return x


class ODEncoder2Decoder(nn.Module):
    # ODEncoder2Decoder 类是一个包装类，将编码器和解码器集成在一起。它定义了编码 (encode)、解码 (decode)、调整输入 (adjust_input)、
    # 调整输出 (adjust_output)、添加噪声 (add_noise) 和前向传播 (forward) 等方法。
    # 在 forward 方法中，先对输入数据添加噪声，然后编码、在潜在空间添加噪声、裁剪、解码，最终输出重构的数据。
    def __init__(
        self,
        in_dim,
        kernel_size=3,
        fold_rate=3,
        input_noise_factor=0.001,
        latent_noise_factor=0.1,
        enc_channel_list=None,
        dec_channel_list=None,
    ):
        super().__init__()
        """
        in_dim: 输入数据的维度。
        kernel_size: 卷积层中使用的核（kernel）的大小。
        fold_rate: 用于确定编码和解码过程中尺寸调整的比率。
        input_noise_factor: 输入数据添加噪声的因子。
        latent_noise_factor: 潜在空间添加噪声的因子。
        enc_channel_list: 编码器各层的通道数量列表。
        dec_channel_list: 解码器各层的通道数量列表。
        """
        self.in_dim = in_dim  # 155160
        self.fold_rate = fold_rate  # 5
        self.kernel_size = kernel_size  # 5
        self.input_noise_factor = input_noise_factor  # 0.001
        self.latent_noise_factor = latent_noise_factor  # 0.1

        # default augment for debug
        if enc_channel_list is None:
            enc_channel_list = [2, 2, 2, 2]  # [4, 4, 4, 4]
        if dec_channel_list is None:
            dec_channel_list = [2, 64, 64, 8]  # [4, 256, 256, 8]

        enc_dim_list = []  # [155625, 31125, 6225, 1245]
        dec_dim_list = []  # [249, 1245, 6225, 31125]

        enc_layer_num = len(enc_channel_list)  # default encoder layer is fixed
        real_input_dim = (
            int(in_dim / self.fold_rate**enc_layer_num + 1) * self.fold_rate**enc_layer_num
        )
        # real_input_dim: 根据fold_rate和编码器层数调整后的真实输入维度。

        for i in range(len(enc_channel_list)):
            dim = real_input_dim // fold_rate**i
            enc_dim_list.append(dim)

        for i in range(len(dec_channel_list)):
            dim = real_input_dim // fold_rate ** (4 - i)
            dec_dim_list.append(dim)

        self.real_input_dim = real_input_dim
        self.encoder = ODEncoder(enc_dim_list, fold_rate, kernel_size, enc_channel_list)
        self.decoder = ODDecoder(dec_dim_list, fold_rate, kernel_size, dec_channel_list)

    # @pysnooper.snoop()
    def encode(self, x, **kwargs):
        x = self.adjust_input(x)
        return self.encoder(x, **kwargs)

    def decode(self, x, **kwargs):
        decoded = self.decoder(x, **kwargs)
        return self.adjust_output(decoded)

    def adjust_output(self, output):
        return output[:, :, : self.in_dim].squeeze(1)

    # @pysnooper.snoop()
    def adjust_input(self, input):
        input_shape = input.shape  # torch.Size([374544])
        # aaa = input.size() # torch.Size([374544]
        # bbb = len(input) # 374544
        # ccc = len(input.size())
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )

        return input

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])
        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x, **kwargs):
        # here is no problem...
        x = self.add_noise(x, self.input_noise_factor)
        # print(x) # tensor([[ 0.2213,  0.4408, -0.1373,  ..., -0.0189, -0.0045,  0.0739]])
        # print(x.shape) # torch.Size([1, 374544])
        # return
        x = self.encode(x)
        x = self.add_noise(x, self.latent_noise_factor)
        x = torch.clamp(x, -1, 1)
        x = self.decode(x)
        return x


# small 和 medium 类分别定义了两种不同配置的模型，继承自 ODEncoder2Decoder 类。
# 它们在初始化时指定了不同的下采样率 fold_rate、卷积核大小 kernel_size、
# 编码器通道数列表 enc_channel_list 和解码器通道数列表 dec_channel_list。


# NOTE: here we define the autoencoder encoder_channel_list
class medium(ODEncoder2Decoder):
    def __init__(
        self,
        in_dim=374544,
        input_noise_factor=0.001,
        latent_noise_factor=0.1,
        fold_rate=5,
        kernel_size=5,
        enc_channel_list=[4, 4, 4, 4],
        dec_channel_list=[4, 256, 256, 8],
    ):
        super().__init__(
            in_dim,
            kernel_size,
            fold_rate,
            input_noise_factor,
            latent_noise_factor,
            enc_channel_list,
            dec_channel_list,
        )


if __name__ == "__main__":
    _ = medium(2048, 0.1, 0.1)

# 如果直接运行这个脚本，将实例化一个 small 模型，输入维度为 2048, 输入噪声系数为 0.1, 潜在空间噪声系数为 0.1。
# 总的来说，这段代码实现了一个基于卷积编码器 - 解码器架构的神经网络模型，用于对一维序列数据进行编码和重构。模型包括一个编码器和一个解码器，
# 分别由多个卷积层和上采样层组成。在编码和解码过程中，还引入了噪声和裁剪操作，以增强模型的鲁棒性和泛化能力。此外，
# 代码还定义了两种不同配置的模型 (small 和 medium), 方便根据实际需求进行选择和调整。
