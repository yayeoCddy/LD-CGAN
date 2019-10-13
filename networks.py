from torchvision import models
import torch
import torch.nn as nn
from functools import partial

class Vgg19(nn.Module):
    def __init__(self,
                 percep_feature=35,
                 use_bn=False,
                 style=False):
        super(Vgg19, self).__init__()
        print('>> load done pretrained Vgg19')
        print('\t exact feature from Vgg19 {}th layer'.format(percep_feature))
        if use_bn:
            model = models.vgg19_bn(pretrained=True)
        else:
            model = models.vgg19(pretrained=True)
        self.per_features = nn.Sequential(*list(model.features.children())[:percep_feature])
        print(self.per_features)
        self.encoder_features = nn.Sequential(*list(model.features.children())[:28])
        if style:
            self.style_features = []
            style_layers = [1, 3, 6, 8, 11]
            for i in style_layers:
                self.style_features.append(nn.Sequential(*list(model.features.children())[:i]))
        # No need to BP to variable
        for param in self.per_features.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.per_features(x)
        return output

    def encoder(self, x):
        output = self.encoder_features(x)
        return output

    def encode_style(self, x):
        output = []
        for i in range(len(self.style_features)):
            output.append(self.style_features[i](x))
        return output


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        model = models.vgg16(pretrained=True)
        # url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        features = model.features
        # print('Load pretrained model from ', url)
        print('load done pretrained Vgg16')
        print('exact feature from Vgg16 relu2-2')

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        # for x in range(9, 16):
        #     self.to_relu_3_3.add_module(str(x), features[x])
        # for x in range(16, 23):
        #     self.to_relu_4_3.add_module(str(x), features[x])

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        # h = self.to_relu_3_3(h)
        # h_relu_3_3 = h
        # h = self.to_relu_4_3(h)
        # h_relu_4_3 = h
        out = h_relu_2_2
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out


class Style_SpatialAttn_Reso2_En2_Decoder2(nn.Module):
    def __init__(self,
                 embedding_dim,
                 noise_dim,
                 hidden_dim,
                 initial_size=4):
        super(Style_SpatialAttn_Reso2_En2_Decoder2, self).__init__()
        # The test_utils is encoded to the text_hidden
        print('>> Init Encoder2')
        sent_dim = 1024
        self.initial_size = initial_size
        # self.encoder2 = Vec2FeatMap(noise_dim + embedding_dim, initial_size, initial_size, hidden_dim)
        self.encoder2 = Vec2FeatMap(noise_dim, initial_size, initial_size, hidden_dim)
        self.text_encoder = CondEmbedding(sent_dim, embedding_dim)
        self.text_disentangle = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, True),
        )

        print('>> Init Decoder2')
        dim = [256, 128, 128, 64, 32]
        print('\tEn2_De2 dim:{}'.format(dim))

        # after adaptiveBN, add a leakyrelu layer
        self.decoder2_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8*8
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, dim[0], 3, 1)
        )
        self.AdBN_8 = AdaptiveBatchNorm(dim[0], embedding_dim)
        self.decoder2_16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8*8
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[0], dim[1], 3, 1)
        )
        self.AdBN_16 = AdaptiveBatchNorm(dim[1], embedding_dim)
        self.decoder2_32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8*8
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[1], dim[2], 3, 1)
        )
        self.AdBN_32 = AdaptiveBatchNorm(dim[2], embedding_dim)
        self.decoder2_64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 8*8
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[2], dim[3], 3, 1)
        )
        self.AdBN_64 = AdaptiveBatchNorm(dim[3], embedding_dim)
        # self.AdBN_64 = AdaptiveBN_Block(dim[3])

        self.fuseBlock_64 = AttensionFuseBlock(dim[2], dim[3])
        self.decoder2_64_toRGB = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[3], 3, 3, 1),
            nn.Tanh()
        )
        self.decoder2_128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128*128
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[3], dim[4], 3, 1)
        )
        self.AdBN_128 = AdaptiveBN_Block(dim[4])
        self.fuseBlock_128 = AttensionFuseBlock(dim[3], dim[4])
        self.decoder2_128_toRGB = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim[4], 3, 3, 1),
            nn.Tanh()
        )

        self.apply(_weight_initializer)

    def forward(self, sent_embeddings, z):
        sent_num = 10.0
        print("sample num:{}".format(sent_num))
        sent_random = 0
        for _ in range(int(sent_num)):
            sent_random_i, mean, logsigma = self.text_encoder(sent_embeddings)
            sent_random += sent_random_i
        sent_random /= sent_num

        lrealu = nn.LeakyReLU(0.2, True)
        # sent_random, mean, logsigma = self.text_encoder(sent_embeddings)
        # concat_feat = torch.cat([sent_random, z], dim=1)
        # z_feat = self.encoder2(concat_feat)
        z_feat = self.encoder2(z)
        sent_disent = self.text_disentangle(sent_random)
        # sent_disent = sent_random
        feat = z_feat
        for i in range(1, int(math.log(128 // self.initial_size, 2)) + 1):
            size = 4 * (2 ** i)
            conv = getattr(self, 'decoder2_{}'.format(size))
            block = getattr(self, 'AdBN_{}'.format(size))
            feat = conv(feat)
            feat = block(feat, sent_disent)
            feat = lrealu(feat)
            if size == 32:
                feat_32 = feat
            if size == 64:
                feat = self.fuseBlock_64(feat_32, feat)
                feat_64 = feat
                img64 = self.decoder2_64_toRGB(feat_64)
            if size == 128:
                feat = self.fuseBlock_128(feat_64, feat)
                img128 = self.decoder2_128_toRGB(feat)
        return img64, img128, z_feat, mean, logsigma


class AdaptiveBatchNorm(nn.Module):
    def __init__(self, in_channel, style_dim, norm='batch'):
        super(AdaptiveBatchNorm, self).__init__()
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(in_channel)
        else:
            self.norm = nn.BatchNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        # self.style.linear.bias.data[:in_channel] = 1
        # self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class AdaptiveBN_Block(nn.Module):
    def  __init__(self, in_channel, kernel_size=3, embedding_dim=128):
        super(AdaptiveBN_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size)
        )

        # self.noise1 = NoiseInjection(in_channel)
        self.adabn1 = AdaptiveBatchNorm(in_channel, embedding_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size)
        )

        # self.noise2 = equal_lr(NoiseInjection(in_channel))
        self.adabn2 = AdaptiveBatchNorm(in_channel, embedding_dim)
        # self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, text):
        out = self.conv1(input)
        # out = self.noise1(out, noise)
        out = self.adabn1(out, text)
        out = self.lrelu1(out)

        out = self.conv2(out)
        # out = self.noise2(out, noise)
        out = self.adabn2(out, text)
        # out = self.lrelu2(out)

        # out = input + out

        return out


class Logits_Convs(torch.nn.Module):
    def __init__(self, in_chanel):
        super(Logits_Convs, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)

        self.encoder1 = conv_norm(in_chanel, in_chanel, norm_layer, activation=activ)
        self.encoder2 = nn.Sequential(conv_norm(in_chanel, in_chanel, norm_layer, activation=activ),
                                      conv_norm(in_chanel, in_chanel // 2, norm_layer, activation=activ))
        self.encoder3 = nn.Sequential(conv_norm(in_chanel // 2, in_chanel // 4, norm_layer, activation=activ),
                                      conv_norm(in_chanel // 4, in_chanel // 4, norm_layer, activation=activ))
        self.logits_1 = nn.Conv2d(in_chanel, 1, 3, padding=1)
        self.logits_2 = nn.Conv2d(in_chanel // 2, 1, 3, padding=1)
        self.logits_3 = nn.Conv2d(in_chanel // 4, 1, 3, padding=1)

        self.apply(_weight_initializer)

    def forward(self, inputs):
        feat_1 = self.encoder1(inputs)
        feat_2 = self.encoder2(feat_1)
        feat_3 = self.encoder3(feat_2)

        return self.logits_1(feat_1), self.logits_2(feat_2), self.logits_3(feat_3)


class Discriminator(torch.nn.Module):
    def __init__(self, num_chan, hid_dim, sent_dim, emb_dim, side_output_at=[64, 128]):
        """
        Parameters:
        ----------
        num_chan: int
            channel of generated images.
        enc_dim: int
            Reduce images inputs to (B, enc_dim, H, W) feature
        emb_dim : int
            the dimension of compressed sentence embedding.
        side_output_at:  list
            contains local loss size for discriminator at scales.
        """

        super(Discriminator, self).__init__()
        self.__dict__.update(locals())

        activ = nn.LeakyReLU(0.2, True)
        norm_layer = partial(nn.BatchNorm2d, affine=True)

        self.side_output_at = side_output_at

        if 64 in side_output_at:  # discriminator for 64 input
            self.img_encoder_64 = ImageDown(64, num_chan, hid_dim)  # 4
            out_dim = hid_dim * 2 * 4
            self.pair_disc_64 = DiscClassifier(out_dim, emb_dim, kernel_size=4)
            self.local_img_disc_64 = nn.Conv2d(out_dim, 1, kernel_size=4, padding=1, bias=True)
            # self.local_img_disc_64 = Logits_Convs(out_dim)
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)
            print('D_64 done.')

        if 128 in side_output_at:  # discriminator for 128 input
            self.img_encoder_128 = ImageDown(128, num_chan, hid_dim)  # 4
            out_dim = hid_dim * 8
            self.pair_disc_128 = DiscClassifier(out_dim, emb_dim, kernel_size=4)
            self.local_img_disc_128 = nn.Conv2d(out_dim, 1, kernel_size=4, padding=1, bias=True)
            # self.local_img_disc_128 = Logits_Convs(out_dim)
            # self.class_output_128 = nn.Sequential(nn.Linear(4 * 4 * out_dim, 201))
            self.class_output_128 = nn.Sequential(nn.Linear(4 * 4 * out_dim, 103))
            # map sentence to a code of length emb_dim
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)
            print('D_128 done.')

        if 256 in side_output_at:  # discriminator for 128 input
            self.img_encoder_256 = ImageDown(256, num_chan, hid_dim)  # 4
            out_dim = hid_dim // 2 * 16
            self.pair_disc_256 = DiscClassifier(out_dim, emb_dim, kernel_size=4)
            self.local_img_disc_256 = nn.Conv2d(out_dim, 1, kernel_size=4, padding=1, bias=True)
            self.class_output_256 = nn.Sequential(nn.Linear(4 * 4 * out_dim, 201))
            # map sentence to a code of length emb_dim
            _layers = [nn.Linear(sent_dim, emb_dim), activ]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)
            print('D_256 done.')

        self.apply(_weight_initializer)

        print('>> Init HDGAN Discriminator')
        print('\t Add adversarial loss at scale {}'.format(str(side_output_at)))

    def forward(self, images, embedding):
        '''
        Parameters:
        -----------
        images:    (B, C, H, W)
            input image tensor
        embedding : (B, sent_dim)
            corresponding embedding
        outptuts:
        -----------
        out_dict: dict
            dictionary containing: pair discriminator output and image discriminator output
        '''
        this_img_size = images.size()[3]
        assert this_img_size in [64, 128, 256], 'wrong input size {} in image discriminator'.format(this_img_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(this_img_size))
        local_img_disc = getattr(self, 'local_img_disc_{}'.format(this_img_size))
        pair_disc = getattr(self, 'pair_disc_{}'.format(this_img_size))
        context_emb_pipe = getattr(
            self, 'context_emb_pipe_{}'.format(this_img_size))

        sent_code = context_emb_pipe(embedding)
        img_code = img_encoder(images)
        local_img_disc_out = local_img_disc(img_code)
        pair_disc_out = pair_disc(sent_code, img_code).view(images.size(0), -1)
        if this_img_size == 128:
            class_output = getattr(self, 'class_output_{}'.format(this_img_size))
            class_out = class_output(img_code.view(images.size()[0], -1))
            return pair_disc_out, local_img_disc_out, class_out
        else:
            return pair_disc_out, local_img_disc_out


class ImageDown(torch.nn.Module):
    def __init__(self, input_size, num_chan, dim):
        """
            Parameters:
            ----------
            input_size: int
                input image size, can be 64, or 128, or 256
            num_chan: int
                channel of input images.
            out_dim : int
                the dimension of generated image code.
        """

        super(ImageDown, self).__init__()
        self.__dict__.update(locals())

        norm_layer = partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)

        _layers = []
        # use large kernel_size at the end to prevent using zero-padding and stride
        if input_size == 64:
            this_dim = dim * 2
            _layers += [conv_norm(num_chan, this_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 32
            _layers += [conv_norm(this_dim, this_dim * 2, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(this_dim * 2, this_dim * 4, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [
                conv_norm(this_dim * 4, this_dim * 4, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)] # 4
            # _layers += [conv_norm(this_dim * 4, this_dim * 4, norm_layer, stride=2, activation=activ)] # 4
            # _layers += [conv_norm(this_dim * 4, this_dim * 4, norm_layer, stride=1, activation=activ)]  # 4
            _layers1 = _layers

        if input_size == 128:
            this_dim = dim
            _layers += [conv_norm(num_chan, this_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 64
            _layers += [conv_norm(this_dim, this_dim * 2, norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(this_dim * 2, this_dim * 4, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(this_dim * 4, this_dim * 8, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [
                conv_norm(this_dim * 8, this_dim * 8, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)] # 4
            # _layers += [conv_norm(this_dim * 4, this_dim * 4, norm_layer, stride=2, activation=activ)] # 4
            # _layers += [conv_norm(this_dim * 4, this_dim * 4, norm_layer, stride=1, activation=activ)]  # 4
            _layers1 = _layers


        if input_size == 256:
            this_dim = dim // 2 # for testing
            _layers += [conv_norm(num_chan, this_dim, norm_layer, stride=2, activation=activ, use_norm=False)]  # 128
            _layers += [conv_norm(this_dim, this_dim * 2, norm_layer, stride=2, activation=activ)]  # 64
            _layers += [conv_norm(this_dim * 2, this_dim * 4, norm_layer, stride=2, activation=activ)]  # 32
            _layers += [conv_norm(this_dim * 4, this_dim * 8, norm_layer, stride=2, activation=activ)]  # 16
            _layers += [conv_norm(this_dim * 8, this_dim * 16, norm_layer, stride=2, activation=activ)]  # 8
            _layers += [ conv_norm(this_dim * 16, this_dim * 16, norm_layer, stride=1, activation=activ, kernel_size=5, padding=0)] # 4
            _layers1 = _layers

        self.node1 = nn.Sequential(*_layers1)

    def forward(self, inputs):
        out1 = self.node1(inputs)
        return out1


class DiscClassifier(nn.Module):
    def __init__(self, enc_dim, emb_dim, kernel_size):
        """
            Parameters:
            ----------
            enc_dim: int
                the channel of image code.
            emb_dim: int
                the channel of sentence code.
            kernel_size : int
                kernel size used for final convolution.
        """

        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())
        norm_layer = partial(nn.BatchNorm2d, affine=True)
        activ = nn.LeakyReLU(0.2, True)
        inp_dim = enc_dim + emb_dim

        _layers = [conv_norm(inp_dim, enc_dim, norm_layer, kernel_size=1, stride=1, activation=activ),
                   nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True)]   # 1

        self.node = nn.Sequential(*_layers)

    def forward(self, sent_code, img_code):
        sent_code = sent_code.view(img_code.size()[0], -1, 1, 1)  # (bs, 128, 1, 1)
        sent_code = sent_code.repeat(1, 1, img_code.size()[-1], img_code.size()[-1])
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)

        return output


class AttensionFuseBlock(nn.Module):
    def __init__(self, low_dim, high_dim, norm=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True)):
        super(AttensionFuseBlock, self).__init__()

        self.up_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(low_dim, high_dim, 3, 1))

        self.mask = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(2 * high_dim, 1, 3, 1),
            nn.Sigmoid())

        self.model_2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(high_dim, high_dim, 3, 1),
            norm(high_dim),
            activation)

    def forward(self, lowx1, highx2):
        x1 = self.up_layer(lowx1)
        x = torch.cat((x1, highx2), 1)
        mask = self.mask(x)
        mask_1, mask_2 = mask, 1 - mask
        x1_mask = x1 * mask_1
        x2_mask = highx2 * mask_2
        x = x1_mask + x2_mask
        out = self.model_2(x)
        return out


class Vec2FeatMap(nn.Module):
    def __init__(self, in_dim, row, col, channel, norm='batch', activ=None):
        super(Vec2FeatMap, self).__init__()
        self.channel = channel
        self.row = row
        self.col = col
        out_dim = row*col*channel
        if norm == 'batch':
            _layers = [nn.Linear(in_dim, out_dim)]
            _layers += [nn.BatchNorm1d(out_dim, affine=True)]
        elif norm == 'instance':
            _layers = [nn.Linear(in_dim, out_dim)]
            _layers += [nn.InstanceNorm1d(out_dim, affine=True)]
        # else:
        #     _layers = [spectral_norm(nn.Linear(in_dim, out_dim))]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


class CondEmbedding(nn.Module):
    def __init__(self, sent_dim, embed_dim):
        super(CondEmbedding, self).__init__()

        self.noise_dim = sent_dim
        self.emb_dim = embed_dim
        self.linear = nn.Linear(sent_dim, embed_dim * 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def sample_encoded_context(self, mean, logsigma):
        epsilon = torch.cuda.FloatTensor(mean.size()).normal_()
        # stddev = torch.exp(logsigma)
        # return epsilon.mul(stddev).add_(mean)

        stddev = torch.exp(0.5 * logsigma)
        return epsilon.mul(stddev).add_(mean)

    def forward(self, inputs):
        '''
        inputs: (B, dim)
        return: mean (B, dim), logsigma (B, dim)
        '''
        out = self.relu(self.linear(inputs))
        mean = out[:, :self.emb_dim]
        log_sigma = out[:, self.emb_dim:]

        c = self.sample_encoded_context(mean, log_sigma)
        return c, mean, log_sigma


class DenseBlock(nn.Module):
    """
        modify from torchgan.layers.ResidualBlock2d
    """
    def __init__(self, dim, kernel_size=3, nonlinearity=None, last_nonlinearity=None):
        super(DenseBlock, self).__init__()
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.layers1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(dim))
        self.layers2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(dim))

        self.last_nonlinearity = last_nonlinearity

    def forward(self, x):
        r"""Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        output1 = self.layers1(x)
        inter_input = output1 + x
        output2 = self.layers2(inter_input)
        inter_output = output2 + inter_input
        output = x + inter_output

        return output if self.last_nonlinearity is None else self.last_nonlinearity(output)


class ResidualBlock(nn.Module):
    """
        modify from torchgan.layers.ResidualBlock2d
    """
    def __init__(self, dim, kernel_size=3, norm='batch', nonlinearity=None, last_nonlinearity=None):
        super(ResidualBlock, self).__init__()
        nl = nn.LeakyReLU(0.2, True) if nonlinearity is None else nonlinearity
        if norm == 'batch':
            norm_layer = partial(nn.BatchNorm2d, affine=True)
        elif norm == 'instance':
            norm_layer = partial(nn.InstanceNorm2d, affine=True)

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=0),
            norm_layer(dim),
            nl,
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=0),
            norm_layer(dim))

        self.last_nonlinearity = last_nonlinearity

    def forward(self, x):
        r"""Computes the output of the residual block

        Args:
            x (torch.Tensor): A 4D Torch Tensor which is the input to the Residual Block.

        Returns:
            4D Torch Tensor after applying the desired functions as specified while creating the
            object.
        """
        output = x + self.layers(x)
        return output if self.last_nonlinearity is None else self.last_nonlinearity(output)


def branch_out(in_dim, out_dim=3):
    _layers = [nn.ReflectionPad2d(1),
               nn.Conv2d(in_dim, out_dim,
                         kernel_size=3, padding=0, bias=False)]
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)


def pad_conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, use_activation=True,
                  use_bias=False, activation=nn.ReLU(True)):
    # designed for generators
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]

    seq += [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=0, bias=use_bias),
            norm_layer(dim_out)]

    if use_activation:
        seq += [activation]

    return nn.Sequential(*seq)


def conv_norm(dim_in, dim_out, norm_layer, kernel_size=3, stride=1, use_activation=True,
              use_bias=False, activation=nn.ReLU(True), use_norm=True, padding=None):
    # designed for discriminator

    if kernel_size == 3:
        padding = 1 if padding is None else padding
    else:
        padding = 0 if padding is None else padding

    seq = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, bias=use_bias, stride=stride),
           ]
    if use_norm:
        seq += [norm_layer(dim_out)]
    if use_activation:
        seq += [activation]

    return nn.Sequential(*seq)


def _weight_initializer(self):
    r"""Default weight initializer for all generator models.
    Models that require custom weight initialization can override this method
    """
    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)