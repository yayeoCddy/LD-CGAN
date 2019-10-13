# -*- coding: utf-8 -*-

import argparse, random
import sys
sys.path.append('..')
from models.networks import Discriminator, Vgg19, Style_SpatialAttn_Reso2_En2_Decoder2
from train_txtdomain import compute_d_pair_loss
from utils.plot_utils import save_images
from fuel.datasets import Dataset
import time
import torch.optim as optim
from utils.utils import *
from tensorboardX import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# for reproducibility
seed = 1024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sys.path.insert(0, os.path.join('..', '..'))
model_root = '/home/chendaiyuan/txt2imgGAN/Img_Models'

def to_img_dict(*inputs):
    if type(inputs[0]) == tuple:
        inputs = inputs[0]
    res = {}
    res['output_64'] = inputs[0]
    res['output_128'] = inputs[1]
    latent_feat = inputs[2]
    mean, logsigma = inputs[3], inputs[4]

    return res, latent_feat, mean, logsigma

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--reuse_weights', action='store_true',
                        default=False, help='continue from last checkout point')
    parser.add_argument('--load_from_epoch', type=int,
                        default=0, help='load from epoch')

    parser.add_argument('--batch_size', type=int,
                        default=16, metavar='N', help='batch size.')
    parser.add_argument('--device_id', type=int,
                        default=0, help='which device')
    parser.add_argument('--gpus', type=str, default='0',
                        help='which gpu')

    parser.add_argument('--model_name', type=str, default='32_128_AE-VAE')
    parser.add_argument('--dataset', type=str, default='birds',
                        help='which dataset to use [birds or flowers]')

    parser.add_argument('--finest_size', type=int, default=64,
                        metavar='N', help='target image size.')
    parser.add_argument('--emb_dim', type=int, default=128,
                        metavar='N', help='vae hidden dimention.')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        metavar='N', help='latent hidden dimention.')
    parser.add_argument('--noise_dim', type=int, default=100,
                        metavar='N', help='noise dimention.')
    parser.add_argument('--G_step_dim', type=int, default=64,
                        help='deciding the depth of G.')
    parser.add_argument('--D_step_dim', type=int, default=32,
                        help='deciding the depth of D.')
    parser.add_argument('--test_sample_num', type=int, default=1,
                        help='The number of runs for each embeddings when testing')
    parser.add_argument('--num_emb', type=int, default=4, metavar='N',
                        help='number of emb chosen for each image during training.')
    parser.add_argument('--label_smooth', type=float, default='0.0',
                        help='whether to use label smooth trick [0.0 or 0.1]')
    parser.add_argument('--instance_noise', action='store_true',
                        help='whether to add instance noise to D')
    parser.add_argument('--reconstruction', type=str, default='L2',
                        help='which loss to use to reconstruct En1_De1 image [L1 or L2]')
    # Instance noise
    # https://github.com/soumith/ganhacks/issues/14#issuecomment-312509518
    # https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    # parser.add_argument('--inst_noise_sigma', type=float, default=0.1)
    # parser.add_argument('--inst_noise_sigma_epoch', type=int, default=200)

    parser.add_argument('--g_lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--d_lr', type=float, default=0.0004, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epoch_decay', type=float, default=200,
                        help='decay learning rate by half every epoch_decay')
    parser.add_argument('--loss_per_epoch', type=int, default=10,
                        help='print losses per epoch')
    parser.add_argument('--loss_per_iter', type=int, default=200,
                        help='print losses per iteration')
    parser.add_argument('--maxepoch', type=int, default=801, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save_freq', type=int, default=200, metavar='N',
                        help='save the model per epoch')
    parser.add_argument('--display_test_imgs', type=int, default=100, metavar='N',
                        help='save test images every {} epochs')

    parser.add_argument('--server_port', type=int, default=10100, metavar='N',
                        help='number of emb chosen for each image during training.')

    # add more
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)

    if args.server_port == 12306:
        data_root = '/hhd12306/chendaiyuan/Data'
    elif args.server_port == 12315:
        data_root = '/hhd/chendaiyuan/Data'
    else:
        data_root = '/data5/chendaiyuan/Data'

    torch.cuda.empty_cache()
    vgg19 = Vgg19().cuda()
    # en2_decoder2 = Style_Reso2_En2_Decoder2(embedding_dim=args.emb_dim, noise_dim=args.noise_dim,
    #                                         hidden_dim=args.hidden_dim).cuda()
    en2_decoder2 = Style_SpatialAttn_Reso2_En2_Decoder2(embedding_dim=args.emb_dim, noise_dim=args.noise_dim,
                                                        hidden_dim=args.hidden_dim).cuda()
    netD = Discriminator(num_chan=3, hid_dim=args.D_step_dim, sent_dim=1024, emb_dim=args.emb_dim,
                         side_output_at=[64, 128]).cuda()
    print(en2_decoder2)
    print(netD)

    gpus = [a for a in range(len(args.gpus.split(',')))]
    torch.cuda.set_device(gpus[0])
    args.batch_size = args.batch_size * len(gpus)

    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = True

    data_name = args.dataset
    datadir = os.path.join(data_root, data_name)

    dataset_train = Dataset(datadir, img_size=[64, 128],
                            batch_size=args.batch_size, n_embed=args.num_emb, mode='train')
    dataset_test = Dataset(datadir, img_size=[64, 128],
                           batch_size=args.batch_size, n_embed=1, mode='test')

    model_name = '{}_{}'.format(args.model_name, data_name)
    writer = SummaryWriter('{}_log'.format(model_name))

    # --------------------- preparing training ----------------------- #
    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' get train and test data sampler '''
    updates_per_epoch = int(dataset_train._num_examples / args.batch_size)

    model_folder = os.path.join(model_root, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # sphere Gaussice noise
    z = torch.FloatTensor(args.batch_size, args.noise_dim)
    z = to_device(z)

    if args.instance_noise:
        inst_noise_std = 0.1

    # ----------------------- reload model------------------------------- #
    if args.reuse_weights:
        premodel_folder = ''
        en2_de2_weights = os.path.join(
            model_folder, 'en2_de2_epoch{}.pth'.format(args.load_from_epoch))
        image_D_weights = os.path.join(
            model_folder, 'netD_epoch{}.pth'.format(args.load_from_epoch))

        print('reload weights from {}'.format(en2_de2_weights))

        if os.path.exists(en2_de2_weights):
            en2_de2_weights_dict = torch.load(en2_de2_weights, map_location=lambda storage, loc: storage)
            en2_decoder2_ = en2_decoder2.module if 'DataParallel' in str(type(en2_decoder2)) else en2_decoder2
            en2_decoder2_.load_state_dict(en2_de2_weights_dict, strict=False)
            image_D_weights_dict = torch.load(image_D_weights, map_location=lambda storage, loc: storage)
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            netD_.load_state_dict(image_D_weights_dict, strict=False)

            # start_epoch = 1
            start_epoch = args.load_from_epoch + 1
            d_lr /= 2 ** (start_epoch // 200)
            g_lr /= 2 ** (start_epoch // 200)

            if args.instance_noise:
                inst_noise_std /= 2 ** (start_epoch // 200)
        else:
            raise ValueError('{} do not exist'.format(en2_de2_weights))
    else:
        start_epoch = 1

    ''' configure optimizer '''
    en2_de2_optim = optim.Adam(filter(lambda p: p.requires_grad, en2_decoder2.parameters()),
                               g_lr, [0.5, 0.999])
    image_D_optim = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()),
                               d_lr, [0.5, 0.999])

    # ------------------------- Start training ------------------------------- #
    print('>> Start training ...')
    en2_decoder2.train()
    netD.train()

    for epoch in range(start_epoch, tot_epoch):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % 200 == 0:
            g_lr *= 0.5
            d_lr *= 0.5

            set_lr(en2_de2_optim, g_lr)
            set_lr(image_D_optim, d_lr)

            if args.instance_noise:
                inst_noise_std *= 0.5

        # reset to prevent StopIteration
        train_sampler = iter(dataset_train)
        test_sampler = iter(dataset_test)

        for it in range(updates_per_epoch):

            # ------------- train image D --------------- #
            try:
                images, wrong_images, np_embeddings, _, np_class_ids, np_wrong_class_ids = \
                    next(train_sampler)
            except:
                train_sampler = iter(dataset_train)  # reset
                images, wrong_images, np_embeddings, _, np_class_ids, np_wrong_class_ids = \
                    next(train_sampler)
            embeddings = to_device(np_embeddings)
            # number of classification, 201 for birds, 103 for flowers, 92 for coco
            one_hot_class_ids = to_device(ori_one_hot(np_class_ids, class_num=103))
            one_hot_wrong_class_ids = to_device(ori_one_hot(np_wrong_class_ids, class_num=103))

            requires_grad(en2_decoder2, False)
            requires_grad(netD, True)
            image_D_optim.zero_grad()

            z.data.normal_(0, 1)
            fake_images, _, _, _ = to_img_dict(en2_decoder2(embeddings, z))

            netD_loss = 0
            dic_D_pair_loss = {}
            dic_D_img_loss = {}
            dic_D_cls_loss = {}
            ''' iterate over image of different sizes.'''
            for key, _ in fake_images.items():
                this_img = to_device(images[key])
                this_wrong = to_device(wrong_images[key])
                this_fake = fake_images[key].detach()

                if args.instance_noise:
                    img_size = this_img.size(-1)
                    this_img += get_instance_noise(args.batch_size, img_size, inst_noise_std)
                    this_wrong += get_instance_noise(args.batch_size, img_size, inst_noise_std)
                    this_fake += get_instance_noise(args.batch_size, img_size, inst_noise_std)

                if key == 'output_128':
                    real_logit, real_img_logit_local, real_cls_logit = netD(this_img, embeddings)
                    wrong_logit, wrong_img_logit_local, wrong_cls_logit = netD(this_wrong, embeddings)
                    fake_logit, fake_img_logit_local, fake_cls_logit = netD(this_fake, embeddings)

                else:
                    real_logit, real_img_logit_local = netD(this_img, embeddings)
                    wrong_logit, wrong_img_logit_local = netD(this_wrong, embeddings)
                    fake_logit, fake_img_logit_local = netD(this_fake, embeddings)

                ''' compute disc pair loss '''
                real_labels, fake_labels = get_labels(real_logit, args.label_smooth)
                pair_loss = compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)
                dic_D_pair_loss['{}'.format(key)] = to_numpy(pair_loss).mean()
                netD_loss += pair_loss

                ''' compute disc image loss '''
                if isinstance(real_img_logit_local, tuple):
                    logit = real_img_logit_local[0]
                    real_labels, fake_labels = get_labels(logit, args.label_smooth)
                else:
                    real_labels, fake_labels = get_labels(real_img_logit_local, args.label_smooth)
                img_loss = (compute_MSE_loss(real_img_logit_local, real_labels) + \
                            compute_MSE_loss(wrong_img_logit_local, real_labels)) * 0.5 + \
                            compute_MSE_loss(fake_img_logit_local, fake_labels)

                dic_D_img_loss['{}'.format(key)] = to_numpy(img_loss).mean()
                netD_loss += img_loss

                if key == 'output_128':
                    ''' compute cls loss '''
                    real_cls_loss = compute_BCE_loss(real_cls_logit, one_hot_class_ids, 1.0)
                    wrong_cls_loss = compute_BCE_loss(wrong_cls_logit, one_hot_wrong_class_ids, 1.0)
                    fake_cls_loss = compute_BCE_loss(fake_cls_logit, one_hot_class_ids, 1.0)
                    cls_loss = ((real_cls_loss + fake_cls_loss) * 0.5 + wrong_cls_loss) * 10.0
                    dic_D_cls_loss['{}'.format(key)] = to_numpy(cls_loss).mean()
                    netD_loss += cls_loss

            netD_loss_val = to_numpy(netD_loss).mean()

            netD_loss.backward()
            image_D_optim.step()
            image_D_optim.zero_grad()

            # -------------- train En2_De2 ---------------- #
            requires_grad(en2_decoder2, True)
            requires_grad(netD, False)
            en2_de2_optim.zero_grad()

            z.data.normal_(0, 1)
            fake_images, concat_feat, mean, logsigma = to_img_dict(en2_decoder2(embeddings, z))

            en2_de2_loss = 0
            dic_G_pair_loss = {}
            dic_G_img_loss = {}
            dic_G_cls_loss = {}
            dic_G_per_loss = {}
            '''Compute gen loss'''
            for key, _ in fake_images.items():
                this_fake = fake_images[key]

                if args.instance_noise:
                    img_size = this_fake.size(-1)
                    this_fake_noise = this_fake + get_instance_noise(args.batch_size, img_size, inst_noise_std)

                if key == 'output_128':
                    fake_pair_logit, fake_img_logit_local, fake_cls_logit = netD(this_fake_noise, embeddings)
                else:
                    fake_pair_logit, fake_img_logit_local = netD(this_fake_noise, embeddings)

                # -- compute pair loss ---
                real_labels, _ = get_labels(fake_pair_logit)
                pair_loss = compute_MSE_loss(fake_pair_logit, real_labels, 1.0)
                dic_G_pair_loss['{}'.format(key)] = to_numpy(pair_loss).mean()
                en2_de2_loss += pair_loss

                # -- compute image loss ---
                if isinstance(fake_img_logit_local, tuple):
                    logit = fake_img_logit_local[0]
                    real_labels, _ = get_labels(logit)
                else:
                    real_labels, _ = get_labels(fake_img_logit_local)
                img_loss = compute_MSE_loss(fake_img_logit_local, real_labels, 1.0)
                dic_G_img_loss['{}'.format(key)] = to_numpy(img_loss).mean()
                en2_de2_loss += img_loss

                if key == 'output_128':
                    # -- compute cls loss ---
                    cls_loss = compute_BCE_loss(fake_cls_logit, one_hot_class_ids, 10.0)
                    dic_G_cls_loss['{}'.format(key)] = to_numpy(cls_loss).mean()
                    en2_de2_loss += cls_loss

                    # -- compute perceptual loss ---
                    this_img = to_device(images[key])
                    per_loss = compute_MSE_loss(vgg19(this_fake), vgg19(this_img), 0.1)
                    dic_G_per_loss['{}'.format(key)] = to_numpy(per_loss).mean()
                    en2_de2_loss += per_loss

                if key != 'output_64':
                    size = int(key[-3:]) // 2
                    # -- compute color loss ----
                    input_1_mean, input_1_var = compute_mean_covariance(this_fake)
                    input_2_mean, input_2_var = compute_mean_covariance(fake_images.get('output_{}'.format(size)).detach())
                    color_loss = compute_MSE_loss(input_1_mean, input_2_mean, 1.0) + \
                                 compute_MSE_loss(input_1_var, input_2_var, 5.0)
                    color_loss_val = to_numpy(color_loss).mean()
                    en2_de2_loss += color_loss

            kl_loss = compute_kl_loss(mean, logsigma, 1.0)
            kl_loss_val = to_numpy(kl_loss).mean()
            en2_de2_loss += kl_loss

            en2_de2_loss.backward()
            en2_de2_optim.step()
            en2_de2_optim.zero_grad()

            # --- visualize train samples----
            if it % args.loss_per_iter == 0:
                print('[epoch %d/%d iter %d/%d]: lr = %.10f g_kl_loss = %.5f color_loss = %.5f '
                      'g_loss = %.5f d_loss= %.5f' %
                      (epoch, tot_epoch, it, updates_per_epoch, g_lr, kl_loss_val, color_loss_val,
                       en2_de2_loss, netD_loss_val))
                sys.stdout.flush()

            if epoch % args.loss_per_epoch == 0:
                writer.add_scalar('kl_loss', kl_loss_val, epoch)
                writer.add_scalar('color_loss', color_loss_val, epoch)
                for key, _ in dic_D_pair_loss.items():
                    writer.add_scalar('d_pair_{}_loss'.format(key), dic_D_pair_loss[key], epoch)
                    writer.add_scalar('d_img_{}_loss'.format(key), dic_D_img_loss[key], epoch)
                    writer.add_scalar('g_pair_{}_loss'.format(key), dic_G_pair_loss[key], epoch)
                    writer.add_scalar('g_img_{}_loss'.format(key), dic_G_img_loss[key], epoch)
                for key, _ in dic_D_cls_loss.items():
                    writer.add_scalar('d_cls_{}_loss'.format(key), dic_D_cls_loss[key], epoch)
                    writer.add_scalar('g_cls_{}_loss'.format(key), dic_G_cls_loss[key], epoch)
                    writer.add_scalar('g_per_{}_loss'.format(key), dic_G_per_loss[key], epoch)

        if epoch % args.display_test_imgs == 0:
            # generate and visualize testing results per epoch
            # display original image and the sampled images
            test_images, _, test_embeddings, _, _, _ = next(test_sampler)
            var_test_embeddings = to_device(test_embeddings)
            vis_samples = {}
            for t in range(args.test_sample_num):
                with torch.no_grad():
                    testing_z = z.data.normal_(0, 1)
                    fake_images, _, _, _ = to_img_dict(en2_decoder2(var_test_embeddings, testing_z))
                    samples = fake_images
                if t == 0:
                    for k in samples.keys():
                        #  +1 to make space for real image
                        vis_samples[k] = [None for i in range(args.test_sample_num + 1)]
                for k, v in samples.items():
                    cpu_data = to_numpy(v)[0:8]
                    if t == 0:
                        if vis_samples[k][0] is None:
                            vis_samples[k][0] = test_images[k][0:8]
                        else:
                            vis_samples[k][0] = np.concatenate([vis_samples[k][0], test_images[k][0:8]], 0)

                    if vis_samples[k][t + 1] is None:
                        vis_samples[k][t + 1] = cpu_data
                    else:
                        vis_samples[k][t + 1] = np.concatenate([vis_samples[k][t + 1], cpu_data], 0)

            # visualize testing samples
            for typ, v in vis_samples.items():
                save_images(v, epoch, 'test_samples_{}'.format(typ), path=model_folder)

        ''' save weights '''
        if epoch % args.save_freq == 0:
            en2_decoder2 = en2_decoder2.cpu()
            en2_decoder2_ = en2_decoder2.module if 'DataParallel' in str(type(en2_decoder2)) else en2_decoder2
            torch.save(en2_decoder2_.state_dict(), os.path.join(
                model_folder, 'en2_de2_epoch{}.pth'.format(epoch)))
            en2_decoder2 = en2_decoder2.cuda()
            netD = netD.cpu()
            netD_ = netD.module if 'DataParallel' in str(type(netD)) else netD
            torch.save(netD_.state_dict(), os.path.join(
                model_folder, 'netD_epoch{}.pth'.format(epoch)))
            netD = netD.cuda()

            print('save weights at {}'.format(model_folder))

        end_timer = time.time() - start_timer
        print(
            'epoch {}/{} finished [time = {}s]'.format(epoch, tot_epoch, end_timer))