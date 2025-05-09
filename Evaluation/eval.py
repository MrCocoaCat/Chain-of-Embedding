import os
import sys
import re
import pickle
import argparse
import numpy as np
from scipy.optimize import minimize

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy import interpolate

#project_root_path = os.environ["PROJECT_PATH"]
#sys.path.append(project_root_path)
project_root_path = "C:\\Users\\liyubo\\Documents\\GitHub\\Chain-of-Embedding\\"
from Data.load_data import DatasetInfo
from config_pool import MODEL_POOL, DATASET_POOL, LANGUAGE_MAPPING
from match import AnswerParsing


class StandardEvaluation:
    def __init__(self, dataset_list):
        self.data_all = []
        self.data_size = 0
        for i, dataset in enumerate(dataset_list):
            data_loader = DatasetInfo(args.dataset)
            self.data_all.extend(data_loader.data)
            self.data_size += data_loader.data_size

    def std_eval(self, args):
        answerparsing = AnswerParsing(args.dataset)
        output_dir = os.path.join(project_root_path, "OutputInfo",args.language,"Output", args.model_name, args.dataset)
        coe_dir = os.path.join(project_root_path, "OutputInfo",args.language,"CoE", args.model_name, args.dataset)
        output_list, coe_list, binary_list = [], [], []
        acc = 0
        for i in range(int(self.data_size)):
            sample = self.data_all[i]
            true_output = sample["answer"]
            with open(os.path.join(output_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                output = pickle.load(file)
            pred_output = output["output_seq"]
            with open(os.path.join(coe_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                coe = pickle.load(file)
            extracted_answer, binary = answerparsing.dataset_parse(pred_output, true_output, sample)
            if binary: acc += 1
            output_list.append(output)
            coe_list.append(coe)
            binary_list.append(binary)
        return round(acc / int(self.data_size), 3), output_list, coe_list, binary_list

    def std_eval(self, args, range_start=0,range_end=None):
        answerparsing = AnswerParsing(args.dataset)
        output_dir = os.path.join(project_root_path, "OutputInfo", args.language, "Output", args.model_name,
                                  args.dataset)
        coe_dir = os.path.join(project_root_path, "OutputInfo", args.language, "CoE", args.model_name, args.dataset)
        output_list, coe_list, binary_list = [], [], []
        acc = 0
        for i in range(range_start,range_end):
            sample = self.data_all[i]
            true_output = sample["answer"]
            with open(os.path.join(output_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                output = pickle.load(file)
            pred_output = output["output_seq"]
            with open(os.path.join(coe_dir, f"{args.dataset}_{str(i)}.pkl"), 'rb') as file:
                coe = pickle.load(file)
            extracted_answer, binary = answerparsing.dataset_parse(pred_output, true_output, sample)
            if binary: acc += 1
            output_list.append(output)
            coe_list.append(coe)
            binary_list.append(binary)
        return round(acc / (range_end-range_start), 3), output_list, coe_list, binary_list



class SelfEvaluation:
    def __init__(self, dataset_list):
        self.data_all = []
        self.data_size = 0
        for i, dataset in enumerate(dataset_list):
            data_loader = DatasetInfo(args.dataset)
            self.data_all.extend(data_loader.data)
            self.data_size += data_loader.data_size

    def self_eval(self, score_list, binary_list):
        if len(score_list) != len(binary_list):
            raise RuntimeError()
        fpr, tpr, thresholds = roc_curve(binary_list, score_list)
        auroc = auc(fpr, tpr)
        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        precision, recall, _ = precision_recall_curve(binary_list, score_list)
        aupr = auc(recall, precision)

        return round(auroc * 100, 2), round(fpr95 * 100, 2), round(aupr * 100, 2),


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct", choices=MODEL_POOL)
    parser.add_argument("--dataset", type=str, default="commonsenseqa", choices=DATASET_POOL)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    stdeval = StandardEvaluation([args.dataset])

    range_start = 0
    range_end = 200
    acc, output_list, coe_list, binary_list = stdeval.std_eval(args,range_end=range_end)
    print(f"# LLM Answer Accuracy: {acc}")

    input_list = [output_list[i]["input_seq"] for i in range(len(output_list))]
    maxprob_list = [output_list[i]["maxprob"] for i in range(len(output_list))]
    # print(maxprob_list)
    ppl_list = [1 / output_list[i]["ppl"] for i in range(len(output_list))]
    # print(ppl_list)
    entropy_list = [1 / output_list[i]["entropy"] for i in range(len(output_list))]
    # print(entropy_list)
    coer_list = [coe_list[i]["R"] for i in range(len(coe_list))]
    # print(coer_list)
    coec_list = [coe_list[i]["C"] for i in range(len(coe_list))]
    # print(coec_list)

    selfeval = SelfEvaluation([args.dataset])
    maxprob_auroc, maxprob_fpr95, maxprob_aupr = selfeval.self_eval(maxprob_list, binary_list)
    ppl_auroc, ppl_fpr95, ppl_aupr = selfeval.self_eval(ppl_list, binary_list)
    entropy_auroc, entropy_fpr95, entropy_aupr = selfeval.self_eval(entropy_list, binary_list)
    coer_auroc, coer_fpr95, coer_aupr = selfeval.self_eval(coer_list, binary_list)
    coec_auroc, coec_fpr95, coec_aupr = selfeval.self_eval(coec_list, binary_list)

    # auroc fpr95 aupr 是三个核心的评价指标
    print(f"{'maxprob_auroc'.rjust(13)}: {maxprob_auroc:.2f}    {'maxprob_fpr95'.rjust(13)}: {maxprob_fpr95:.2f}    {'maxprob_aupr'.rjust(13)}: {maxprob_aupr:.2f}")
    print(f"{'ppl_auroc'.rjust(13)}: {ppl_auroc:.2f}    {'ppl_fpr95'.rjust(13)}: {ppl_fpr95:.2f}    {'ppl_aupr'.rjust(13)}: {ppl_aupr:.2f}")
    print(f"{'entropy_auroc'.rjust(13)}: {entropy_auroc:.2f}    {'entropy_fpr95'.rjust(13)}: {entropy_fpr95:.2f}    {'entropy_aupr'.rjust(13)}: {entropy_aupr:.2f}")
    print(f"{'coer_auroc'.rjust(13)}: {coer_auroc:.2f}    {'coer_fpr95'.rjust(13)}: {coer_fpr95:.2f}    {'coer_aupr'.rjust(13)}: {coer_aupr:.2f}")
    print(f"{'coec_auroc'.rjust(13)}: {coec_auroc:.2f}    {'coec_fpr95'.rjust(13)}: {coec_fpr95:.2f}    {'coec_aupr'.rjust(13)}: {coec_aupr:.2f}")

    print("**************   base   ************************************")
    file_path = 'E:\\GitHub\\Chain-of-Embedding\\OutputImg\\Qwen2.5-7B-Instruct\\commonsenseqa\\base_list.pickle'
    # 以二进制写入模式打开文件
    with open(file_path, 'rb') as file:
        # 使用 pickle.load 方法从文件中读取对象
        bash_list = pickle.load(file)

    #input_list = [bash_list[i]["input_seq"] for i in range(len(output_list))]
    bash_list = bash_list['base_list']
    maxprob_list = [bash_list[i]["maxprob"] for i in range(range_start,range_end)]
    print("maxprob_list",maxprob_list)
    ppl_list = [1 / bash_list[i]["ppl"] for i in range(range_start,range_end)]
    print("ppl_list",ppl_list)
    entropy_list = [1 / bash_list[i]["entropy"] for i in range(range_start,range_end)]
    print("entropy_list",entropy_list)
    coer_list = [bash_list[i]["R"] for i in range(range_start,range_end)]
    print("coer_list",coer_list)
    coec_list = [bash_list[i]["C"] for i in range(range_start,range_end)]
    print("coec_list",coec_list)

    manhattan_distance_ave = [bash_list[i]["manhattan_distance_ave"]for i in range(range_start,range_end)]
    print("manhattan_distance_ave", manhattan_distance_ave)
    chebyshev_distance_ave = [bash_list[i]["chebyshev_distance_ave"] for i in range(range_start,range_end)]
    print("chebyshev_distance_ave", chebyshev_distance_ave)
    mag_list = [bash_list[i]["Mag"] for i in range(range_start,range_end)]
    print("mag_list", mag_list)
    ang_list = [bash_list[i]["Ang"] for i in range(range_start,range_end)]
    print("ang_list", ang_list)
    norm_2_list_ave = [bash_list[i]["norm_2_list_ave"] for i in range(range_start,range_end)]
    print("norm_2_list_ave", norm_2_list_ave)
    norm_3_list_ave = [bash_list[i]["norm_3_list_ave"] for i in range(range_start,range_end)]
    print("norm_3_list_ave", norm_3_list_ave)

    norm_3_list_ave_n = [bash_list[i]["norm_3_list_ave_n"] for i in range(range_start,range_end)]
    print("norm_3_list_ave_n", norm_3_list_ave_n)

    ssim_matrix_list = [bash_list[i]["al_SSIM_diff_ave1"] for i in range(range_start,range_end)]
    print("ssim_matrix_list", ssim_matrix_list)
    al_SSIM_diff_ave1_x = [bash_list[i]["al_SSIM_diff_ave1_x"] for i in range(range_start,range_end)]
    print("al_SSIM_diff_ave1_x", al_SSIM_diff_ave1_x)

    score_pic_ave = [bash_list[i]["score_pic_ave"] for i in range(range_start,range_end)]
    print("score_pic_ave", score_pic_ave)

    score_pic_var = [bash_list[i]["score_pic_var"] for i in range(range_start,range_end)]
    print("score_pic_var", score_pic_var)

    binary_list = binary_list[range_start:range_end]


    selfeval = SelfEvaluation([args.dataset])

    print("**************   base   ************************************")

    score_pic_ave_auroc,score_pic_ave_fpr95, score_pic_ave_aupr = selfeval.self_eval(score_pic_ave, binary_list)
    print(
        f"{'score_pic_ave_auroc'.rjust(13)}: {score_pic_ave_auroc:.2f}    {'score_pic_ave_fpr95'.rjust(13)}: {score_pic_ave_fpr95:.2f}  {'score_pic_ave_aupr'.rjust(13)}: {score_pic_ave_aupr:.2f}")

    score_pic_var_auroc, score_pic_var_fpr95, score_pic_var_aupr = selfeval.self_eval(score_pic_var, binary_list)
    print(
        f"{'score_pic_var_auroc'.rjust(13)}: {score_pic_var_auroc:.2f}    {'score_pic_var_fpr95'.rjust(13)}: {score_pic_var_fpr95:.2f}  {'score_pic_var_aupr'.rjust(13)}: {score_pic_var_aupr:.2f}")


    ssim_matrix_auroc, ssim_matrix_fpr95, ssim_matrix_aupr = selfeval.self_eval(ssim_matrix_list, binary_list)
    print( f"{'ssim_matrix_auroc'.rjust(13)}: {ssim_matrix_auroc:.2f}    {'ssim_matrix_fpr95'.rjust(13)}: {ssim_matrix_fpr95:.2f}  {'ssim_matrix_aupr'.rjust(13)}: {ssim_matrix_aupr:.2f}")
    al_SSIM_diff_ave1_x_auroc, al_SSIM_diff_ave1_x_fpr95, al_SSIM_diff_ave1_x_aupr = selfeval.self_eval(al_SSIM_diff_ave1_x,                                                                                           binary_list)
    print( f"{'al_SSIM_diff_ave1_x_auroc'.rjust(13)}: {al_SSIM_diff_ave1_x_auroc:.2f}    {'al_SSIM_diff_ave1_x_fpr95'.rjust(13)}: {al_SSIM_diff_ave1_x_fpr95:.2f}   {'al_SSIM_diff_ave1_x_aupr'.rjust(13)}: {al_SSIM_diff_ave1_x_aupr:.2f}")

    norm_3_list_ave_n_auroc,norm_3_list_ave_n_fpr95, norm_3_list_ave_n_aupr = selfeval.self_eval(norm_3_list_ave_n, binary_list)
    print(f"{'norm_3_list_ave_n_auroc'.rjust(13)}: {norm_3_list_ave_n_auroc:.2f}    {'norm_3_list_ave_n_fpr95'.rjust(13)}: {norm_3_list_ave_n_fpr95:.2f}   {'norm_3_list_ave_n_aupr'.rjust(13)}: {norm_3_list_ave_n_aupr:.2f}")


    # auroc fpr95 aupr 是三个核心的评价指标
    maxprob_auroc, maxprob_fpr95, maxprob_aupr = selfeval.self_eval(maxprob_list, binary_list)
    print( f"{'maxprob_auroc'.rjust(13)}: {maxprob_auroc:.2f}    {'maxprob_fpr95'.rjust(13)}: {maxprob_fpr95:.2f} {'maxprob_aupr'.rjust(13)}: {maxprob_aupr:.2f}")
    ppl_auroc, ppl_fpr95, ppl_aupr = selfeval.self_eval(ppl_list, binary_list)
    print( f"{'ppl_auroc'.rjust(13)}: {ppl_auroc:.2f}    {'ppl_fpr95'.rjust(13)}: {ppl_fpr95:.2f}  {'ppl_aupr'.rjust(13)}: {ppl_aupr:.2f}")
    entropy_auroc, entropy_fpr95, entropy_aupr = selfeval.self_eval(entropy_list, binary_list)
    print( f"{'entropy_auroc'.rjust(13)}: {entropy_auroc:.2f}    {'entropy_fpr95'.rjust(13)}: {entropy_fpr95:.2f} {'entropy_aupr'.rjust(13)}: {entropy_aupr:.2f}")



    mag_auroc, mag_fpr95, mag_aupr = selfeval.self_eval(mag_list, binary_list)
    print(  f"{'mag_auroc'.rjust(13)}: {mag_auroc:.2f}    {'mag_fpr95'.rjust(13)}: {mag_fpr95:.2f}  {'mag_aupr'.rjust(13)}: {mag_aupr:.2f}")
    ang_auroc, ang_fpr95, ang_aupr = selfeval.self_eval(ang_list, binary_list)
    print(  f"{'ang_auroc'.rjust(13)}: {ang_auroc:.2f}    {'ang_fpr95'.rjust(13)}: {ang_fpr95:.2f} {'ang_aupr'.rjust(13)}: {ang_aupr:.2f}")
    coer_auroc, coer_fpr95, coer_aupr = selfeval.self_eval(coer_list, binary_list)
    print(f"{'coer_auroc'.rjust(13)}: {coer_auroc:.2f}  {'coer_fpr95'.rjust(13)}: {coer_fpr95:.2f}  {'coer_aupr'.rjust(13)}: {coer_aupr:.2f}")
    coec_auroc, coec_fpr95, coec_aupr = selfeval.self_eval(coec_list, binary_list)
    print( f"{'coec_auroc'.rjust(13)}: {coec_auroc:.2f} {'coec_fpr95'.rjust(13)}: {coec_fpr95:.2f} {'coec_aupr'.rjust(13)}: {coec_aupr:.2f}")

    manhattan_auroc, manhattan_fpr95, manhattan_aupr = selfeval.self_eval(manhattan_distance_ave, binary_list)
    print(f"{'manhattan_auroc'.rjust(13)}: {manhattan_auroc:.2f} {'manhattan_fpr95'.rjust(13)}: {manhattan_fpr95:.2f} {'manhattan_aupr'.rjust(13)}: {manhattan_aupr:.2f}")
    chebyshev_auroc, chebyshev_fpr95, chebyshev_aupr = selfeval.self_eval(chebyshev_distance_ave, binary_list)
    print(f"{'chebyshev_auroc'.rjust(13)}: {chebyshev_auroc:.2f} {'chebyshev_fpr95'.rjust(13)}: {chebyshev_fpr95:.2f} {'chebyshev_aupr'.rjust(13)}: {chebyshev_aupr:.2f}")

