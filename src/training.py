import os
import torch
import numpy as np
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, check_drug_list
import time
from collections import defaultdict
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import multilabel_margin_loss
import math
import dill


def eval_one_epoch(model, data_eval, voc_size, drug_data):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output, _ = model(
                patient_data=input_seq[:adm_idx + 1],
                **drug_data
            )
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )


        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step + 1, len(data_eval)))

    ddi_rate = ddi_rate_score(smm_record, drug_data['ddi_A_final'])
    output_str = '\nDDI Rate: {:.4f}, Jaccard: {:.4f}, PRAUC: {:.4f}, ' +\
        'AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'
    llprint(output_str.format(
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p),
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
    ))
    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), \
        np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt


def Test(model, model_path, device, data_test, voc_size, drug_data):
    with open(model_path, 'rb') as Fin:
        # best_model_dict = torch.load(Fin, map_location=device)
        # model_dict = model.state_dict()
        # best_model_dict = {k: v for k, v in best_model_dict.items() if k in model_dict}
        # model_dict.update(best_model_dict)
        # torch.save(model_dict, '../saved/model_saved/best_model.model')
        model.load_state_dict(torch.load(Fin, map_location=device))
        # model.load_state_dict(model_dict)
    model = model.to(device).eval()
    print('--------------------Begin Testing--------------------')
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
    np.random.seed(0)
    for _ in range(10):
        test_sample = np.random.choice(data_test, sample_size, replace=True)
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, test_sample, voc_size, drug_data)
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)
    metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
    outstring = ''.join([
        "{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
        for idx, (m, s) in enumerate(zip(mean, std))
    ])
    print(outstring)
    print('average test time: {}'.format((time.time() - tic) / 10))
    print('parameters', get_n_params(model))

def evaluate_metrics(predictions, targets, batch=None, val=False):
    metrics = {}
    metrics[f'mean_pred'] = torch.mean(predictions).item()
    metrics[f'std_pred'] = torch.std(predictions).item()
    metrics[f'mean_targets'] = torch.mean(targets).item()
    metrics[f'std_targets'] = torch.std(targets).item()

    return metrics

def Pre_Train(model, device, drug_data, optimizer, scheduler, log_dir, EPOCH=50):
    history = defaultdict(list)
    history_name = 'pre_train_{}.pkl'.format(time.strftime('%Y-%m-%d_%H:%M:%S'))

    best_model, best_loss1, best_loss2, best_epoch = None, None, None, 0

    for epoch in range(EPOCH):
        if epoch % 50 == 0:
            print(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        tic, losses_epoch = time.time(), []
        loss1, loss2 = model(drug_data['mol_data'], drug_data['drug_geo_list'], drug_data['substruct_data'], drug_data['drug_sub_list'])

        loss = loss1+loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print('[Epoch %d; %s: loss1: %.7f loss2: %.7f]' % (epoch, 'train', loss1.item(), loss2.item()))

        if best_loss1 is None and best_loss2 is None:
            best_loss1, best_loss2 = loss1, loss2
        else:
            if loss1 < best_loss1 and loss2 < best_loss2:
                best_loss1, best_loss2 = loss1, loss2
                best_model = model.state_dict()
                best_epoch = epoch
    
    print('[best_Epoch %d; %s: best_loss1: %.7f best_loss2: %.7f]' % (best_epoch, 'final', best_loss1.item(), best_loss2.item()))

    torch.save(best_model, os.path.join(log_dir, 'pretrained_model.pt'))
    
    return model



def Train(model, pre_train_model, device, data_train, data_eval, voc_size, drug_data, optimizer, scheduler, regular, log_dir, coef, target_ddi, target_ja, EPOCH=50):

    pretrained_params = {name: param for name, param in pre_train_model.named_parameters() if 'global_encoder' in name}
    pretrained_sub_params = {name: param for name, param in pre_train_model.named_parameters() if 'substruct_encoder' in name}
    target_param_names = [name for name, _ in model.named_parameters() if 'global_encoder' in name]
    target_sub_param_names = [name for name, _ in model.named_parameters() if 'substruct_encoder' in name]
    with torch.no_grad():  
        for param_name in target_param_names:  
            if param_name in pretrained_params:  
                model.state_dict()[param_name].copy_(pretrained_params[param_name])  
            else:  
                print(f"Warning: {param_name} is not present in the pretrained model.")
        for param_name in target_sub_param_names:  
            if param_name in pretrained_sub_params:  
                model.state_dict()[param_name].copy_(pretrained_sub_params[param_name])  
            else:  
                print(f"Warning: {param_name} is not present in the pretrained model.")  
  
    source_coef = coef
    history, best_epoch, best_ja = defaultdict(list), 0, 0
    total_train_time, ddi_losses, ddi_values = 0, [], []
    history_name = 'history_{}.pkl'.format(time.strftime('%Y-%m-%d_%H:%M:%S'))
    for epoch in range(EPOCH):
        print(f'----------------Epoch {epoch + 1}------------------')
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):

                result, loss_ddi = model(
                    patient_data=input_seq[:adm_idx + 1],
                    **drug_data
                )


                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1
                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], drug_data['ddi_A_final']
                )

                if current_ddi_rate <= target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = coef * (1 - (current_ddi_rate / target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = beta * (0.95 * loss_bce + 0.05 * loss_multi) \
                        + (1 - beta) * loss_ddi

                loss += regular(model)

                ddi_losses_epoch.append(loss_ddi.detach().cpu().item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))
        scheduler.step()
        ddi_losses.append(sum(ddi_losses_epoch) / len(ddi_losses_epoch))
        print(f'\nddi_loss : {ddi_losses[-1]}\n')
        train_time, tic = time.time() - tic, time.time()
        total_train_time += train_time

        print(f'training time: {train_time}, testing time: {time.time() - tic}')

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = \
            eval_one_epoch(model, data_eval, voc_size, drug_data)
        if ja > target_ja:
            coef = coef * 1.1
        else:
            coef = coef * 0.9 if coef * 0.9 > source_coef else source_coef
        print(f'\nnext_coef : {coef}\n')
        ddi_values.append(ddi_rate)
        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['prauc'][-5:])
            ))

        model_name = 'Epoch_{}_TARGET_{:.2f}_JA_{:.4f}_DDI_{:.4f}.model'.format(
            epoch, target_ddi, ja, ddi_rate
        )
        torch.save(model.state_dict(), os.path.join(log_dir, model_name))
        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
        print('best_epoch: {}'.format(best_epoch))
        with open(os.path.join(log_dir, 'best.txt'), 'a') as Fout:
            Fout.write(f'{best_epoch}\n')
        with open(os.path.join(log_dir, 'ddi_losses.txt'), 'w') as Fout:
            for dloss, dvalue in zip(ddi_losses, ddi_values):
                Fout.write(f'{dloss}\t{dvalue}\n')

        with open(os.path.join(log_dir, history_name), 'wb') as Fout:
            dill.dump(history, Fout)
    print('avg training time/epoch: {:.4f}'.format(total_train_time / EPOCH))
