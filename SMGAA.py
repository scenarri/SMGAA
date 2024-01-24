from torchvision import datasets
import argparse
import torch.backends.cudnn as cudnn
import json
from utils import *

def main(args):
    set_seed(729)
    data_root = os.path.abspath(os.path.join(os.getcwd()))

    def getindice(ary, indice):
        a = ''
        for i in range(args.nb_asc):
            a += str(int(ary[indice][i]))
            a += '-'
        return a

    try:
        json_file = open('./class_indices.json', 'rb')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    index = 0


    val_dataset = datasets.ImageFolder(root="./dataset/testset")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    lce = nn.CrossEntropyLoss()

    print("==> Loading data and model...")
    testModel = WrapperModel(load_models(args.model), size=(88 if args.model == 'aconv' else 224))
    cudnn.benchmark = True

    iter = 0
    sucessatk = 0

    if args.SAVE:
        path = data_root + "/Record/"
        folder = path + args.model + '-B' + str(args.popbatch * args.restart) + '-N' + str(args.nb_asc) + '-I' + str(args.maxiter)
        Mkdir(folder)
        indexnote = folder + '/index.txt'
        if os.path.isfile(indexnote):
            indextxt = open(indexnote)
            index = int(indextxt.read())
            indextxt.close()
        sucessatknote = folder + '/sucessatk.txt'
        if os.path.isfile(sucessatknote):
            sucessatktxt = open(sucessatknote)
            sucessatk = int(sucessatktxt.read())
            sucessatktxt.close()

    for comp, target in val_loader:
        iter += 1
        if iter <= index:
            continue

        set_seed(iter)

        data = comp[0][0].cuda()
        seg = comp[0][1][0][0].cuda()
        target = target.cuda()
        gt_label = target.item()

        flag = 0
        lastconf_ret = 1.0
        targetexpand = target.expand(args.popbatch)
        for rest in range(args.restart):
            if flag == 1:
                break

            theta, idx, type = init_theta(args.nb_asc, args.popbatch, seg)

            STEPSIZE = stepsize.expand(args.popbatch, args.nb_asc, -1).clone()
            ADVVEC = torch.zeros_like(STEPSIZE)
            for k in range(args.popbatch):
                for kk in range(args.nb_asc):
                    STEPSIZE[k][kk] =STEPSIZE[k][kk] * torch.from_numpy(adjV(type[k][kk])).cuda()
                    ADVVEC[k][kk] = torch.from_numpy(adjV(type[k][kk])).cuda()

            lastconf = torch.softmax(testModel(data),1)[:,target].expand(args.popbatch,-1)
            lasttheta = theta.detach().clone()

            for attak_iters in range(args.maxiter):
                delta, theta = generateima(theta, 1.)
                delta88 = crop_center(delta,88)
                advdata = torch.clamp(data + delta88, 0, 1.)


                output = testModel(advdata)

                _, flag, min_conf, indice, currentconf = check_attack(output, target)

                print("\rCurrent Conf.{} Suc.: {} ({}/{}) Model: {} nbASC: {}, Batch {}, Iter {}".format(round(min_conf.item(),5), round(sucessatk/iter,5), sucessatk, iter, args.model, args.nb_asc, args.popbatch*args.restart, args.maxiter), end="")
                if flag == 1 or attak_iters == (args.maxiter-1):
                    sucess, _, _, indice, currentconf = check_attack(output, target, last=True)
                    sucessatk += sucess

                    if args.SAVE:
                        indexfile = open(folder + '/index.txt', mode='w')
                        indexfile.write(str(iter))
                        indexfile.close()

                        sucessatkfile = open(folder + '/sucessatk.txt', mode='w')
                        sucessatkfile.write(str(sucessatk))
                        sucessatkfile.close()

                        pert = delta88[indice][0].detach().cpu().numpy()
                        advimg = advdata[indice][0].detach().cpu().numpy()
                        img = data[0][0].detach().cpu().numpy()

                        final_label = output.argmax(1)[indice].item()
                        class_folder = folder + '/detailed/' + str(class_indict[str(gt_label)])
                        Mkdir(class_folder)
                        name = str(class_indict[str(final_label)]) + '-' + str(round(torch.softmax(output,1).max(1).values[indice].item(), 4)) + '-' + str(iter)

                        cv2.imwrite(class_folder + '/' + name + '-ori.jpg', img * 255)
                        cv2.imwrite(class_folder + '/' + name + '.jpg', advimg * 255)
                        cv2.imwrite(class_folder + '/' + name + '-pert.jpg', pert * 255)

                        np.savetxt(class_folder + '/' + name +'-type' + getindice(idx,indice) + '.txt', theta[indice].detach().cpu().numpy(), delimiter=" ")

                        class_folder_img = folder + '/advimg/' + str(class_indict[str(gt_label)])
                        Mkdir(class_folder_img)
                        torch.save(advdata[indice][0].detach().cpu(), class_folder_img + '/' + name + '.pt')
                    break

                celoss = lce(output, targetexpand)
                grad = torch.autograd.grad(celoss, theta, create_graph=False, retain_graph=False)[0]

                dtheta = torch.abs(ADVVEC * (torch.randn_like(STEPSIZE) * (EPS_max-EPS_min)/200 + STEPSIZE)) #200
                theta = theta + dtheta * torch.sign(grad)

                theta = torch.clamp(theta, EPS_min, EPS_max).detach()

                minosconf = currentconf - lastconf
                min_conf_idx = torch.nonzero((minosconf > 0)* 1.)
                if min_conf_idx.shape[0] != 0:
                    for j in range(min_conf_idx.shape[0]):
                        if torch.rand([1]).item() < 0.5:
                            theta[min_conf_idx[j][0]] = lasttheta[min_conf_idx[j][0]]

                max_conf_idx = torch.nonzero((minosconf < 0) * 1.)
                if max_conf_idx.shape[0] != 0:
                    for jj in range(max_conf_idx.shape[0]):
                        STEPSIZE[max_conf_idx[jj][0]] = (STEPSIZE[max_conf_idx[jj][0]] + torch.abs(dtheta[max_conf_idx[jj][0]]))/2

                lastconf = currentconf
                lasttheta = theta.detach().clone()

            if min_conf.item() <= lastconf_ret:
                lastconf_ret = min_conf.item()
                lastadvdata = advdata[indice][0].clone()

    print()
    print("\rCurrent Conf.{} Suc.: {} ({}/{}) Model: {} nbASC: {}, Batch {}, Iter {}".format(round(min_conf.item(), 5),
                                                                                             round(sucessatk / iter, 5),
                                                                                             sucessatk, iter, args.model,
                                                                                             args.nb_asc,
                                                                                             args.popbatch * args.restart,
                                                                                             args.maxiter), end="")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASCAttack for SAR imagery')

    parser.add_argument('--model', type=str, default='aconv', help='target model')
    parser.add_argument('--nb_asc', default=2, type=int, help='number of ASCs that can be added.')
    parser.add_argument('--restart', default=1, type=int, help='maximum number of iteration of restart.')
    parser.add_argument('--maxiter', default=90, type=int, help='maximum interation')
    parser.add_argument('--popbatch', default=100, type=int, help='population')
    #parser.add_argument('--supress', default=1., type=float)
    parser.add_argument('--SAVE', action = 'store_true', help = 'using local surrogates')

    args = parser.parse_args()

    main(args)

