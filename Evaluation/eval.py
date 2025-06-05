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
project_root_path = "E:\\GitHub\\Chain-of-Embedding\\"
from Data.load_data import DatasetInfo
from config_pool import MODEL_POOL, DATASET_POOL, LANGUAGE_MAPPING
from match import AnswerParsing
from sklearn.naive_bayes import GaussianNB


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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

    def min_max_transform(self,score_list):
        scores = np.array(score_list)
        min_val = scores.min()
        max_val = scores.max()
        return (scores - min_val) / (max_val - min_val)

    def self_eval(self, score_list, binary_list):
        if len(score_list) != len(binary_list):
            raise RuntimeError()
        #score_list = self.min_max_transform(score_list)
        fpr, tpr, thresholds = roc_curve(binary_list, score_list)
        auroc = auc(fpr, tpr)
        fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
        precision, recall, _ = precision_recall_curve(binary_list, score_list)
        aupr = auc(recall, precision)
        return round(auroc * 100, 2), round(fpr95 * 100, 2), round(aupr * 100, 2), fpr, tpr, thresholds



def svm_train(source_list,lable):

    # 示例数据
    X = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ])
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(source_list, lable, test_size=0.3, random_state=42)
    # 训练SVM模型
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.2f}")
    return clf

def LR(X_train,y_train):
    # 创建逻辑回归模型
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="eval")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct", choices=MODEL_POOL)
    parser.add_argument("--dataset", type=str, default="commonsenseqa", choices=DATASET_POOL)
    parser.add_argument("--language", type=str, default="en")
    args = parser.parse_args()
    stdeval = StandardEvaluation([args.dataset])

    range_start = 0
    range_end = 100
    acc, output_list, coe_list, binary_list = stdeval.std_eval(args,range_start=range_start,range_end=range_end)
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
    maxprob_auroc, maxprob_fpr95, maxprob_aupr, maxprob_fpr, maxprob_tpr, tmaxprob_hresholds= selfeval.self_eval(maxprob_list, binary_list)
    ppl_auroc, ppl_fpr95, ppl_aupr, ppl_fpr, ppl_tpr, ppl_thresholds = selfeval.self_eval(ppl_list, binary_list)
    entropy_auroc, entropy_fpr95, entropy_aupr ,entropy_fpr, entropy_tpr, entropy_thresholds= selfeval.self_eval(entropy_list, binary_list)
    coer_auroc, coer_fpr95, coer_aupr,coer_fpr, coer_tpr, coer_thresholds = selfeval.self_eval(coer_list, binary_list)
    coec_auroc, coec_fpr95, coec_aupr, coec_fpr, coec_tpr, coec_thresholds = selfeval.self_eval(coec_list, binary_list)
    print(
        f"{'maxprob_auroc'.rjust(30)}: {maxprob_auroc:.2f}{'maxprob_fpr95'.rjust(30)}: {maxprob_fpr95:.2f}{'maxprob_aupr'.rjust(30)}: {maxprob_aupr:.2f}  ")
    print(
        f"{'ppl_auroc'.rjust(30)}: {ppl_auroc:.2f}{'ppl_fpr95'.rjust(30)}: {ppl_fpr95:.2f}{'ppl_aupr'.rjust(30)}: {ppl_aupr:.2f} ")
    print(
        f"{'entropy_auroc'.rjust(30)}: {entropy_auroc:.2f}{'entropy_fpr95'.rjust(30)}: {entropy_fpr95:.2f}{'entropy_aupr'.rjust(30)}: {entropy_aupr:.2f} ")
    print(
        f"{'coer_auroc'.rjust(30)}: {coer_auroc:.2f}{'coer_fpr95'.rjust(30)}: {coer_fpr95:.2f}{'coer_aupr'.rjust(30)}: {coer_aupr:.2f}")
    print(
        f"{'coec_auroc'.rjust(30)}: {coec_auroc:.2f}{'coec_fpr95'.rjust(30)}: {coec_fpr95:.2f}{'coec_aupr'.rjust(30)}: {coec_aupr:.2f}")




    file_path = 'E:\\GitHub\\Chain-of-Embedding\\LayerState\\Qwen2.5-7B-Instruct\\commonsenseqa\\base_list.pickle'
    # 以二进制写入模式打开文件
    with open(file_path, 'rb') as file:
        # 使用 pickle.load 方法从文件中读取对象
        bash_list = pickle.load(file)

    bash_list = bash_list['base_list']
    entropy_list = [1 / bash_list[i]["entropy"] for i in range(range_start,range_end)]
   # R = [bash_list[i]["R"] for i in range(range_start, range_end)]
    binary_list = binary_list[range_start:range_end]
    selfeval = SelfEvaluation([args.dataset])


    # auroc fpr95 aupr 是三个核心的评价指标

    print("****************************************************************   base   ****************************************************************")




    maxprob_list = [bash_list[i]["maxprob"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(maxprob_list, binary_list)
    print(f"maxprob:  {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    entropy_list = [1 / bash_list[i]["entropy"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(entropy_list, binary_list)
    print(f"entropy:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    ppl_list = [1 / bash_list[i]["ppl"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(ppl_list, binary_list)
    print(f"ppl:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    coer_list = [bash_list[i]["R"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(coer_list, binary_list)
    print(f"coe-R:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    coec_list = [bash_list[i]["C"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(coec_list, binary_list)
    print(f"coe-C:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")


    mag_list = [bash_list[i]["Mag"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(mag_list, binary_list)
    print(f"Mag:     {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_linear = [bash_list[i]["Mag_linear"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_linear, binary_list)
    print(f"Mag_linear:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_exp = [bash_list[i]["Mag_exp"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_exp, binary_list)
    print(f"Mag_exp:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_step = [bash_list[i]["Mag_step"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_step, binary_list)
    print(f"Mag_step:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_log = [bash_list[i]["Mag_log"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_log, binary_list)
    print(f"Mag_log:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_reciprocal = [bash_list[i]["Mag_reciprocal"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_reciprocal, binary_list)
    print(
        f"Mag_reciprocal:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Mag_quantile = [bash_list[i]["Mag_quantile"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Mag_quantile, binary_list)
    print(
        f"Mag_quantile:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    ang_list = [bash_list[i]["Ang"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(ang_list, binary_list)
    print(f"Ang:     {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")




    Ang_linear = [bash_list[i]["Ang_linear"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_linear, binary_list)
    print(f"Ang_linear:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Ang_exp = [bash_list[i]["Ang_exp"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_exp, binary_list)
    print(f"Ang_exp:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Ang_step = [bash_list[i]["Angg_step"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_step, binary_list)
    print(f"Ang_step:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Ang_log = [bash_list[i]["Ang_log"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_log, binary_list)
    print(f"Ang_log:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Ang_reciprocal = [bash_list[i]["Ang_reciprocal"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_reciprocal, binary_list)
    print(
        f"Ang_reciprocal:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    Ang_quantile = [bash_list[i]["Ang_quantile"] for i in range(range_start, range_end)]
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(Ang_quantile, binary_list)
    print(
        f"Ang_quantile:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")


    print("利用MAG 和 ANG 进行计算 ")
    coe_R_1 = np.array(mag_list) - np.array(ang_list)
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(coe_R_1, binary_list)
    print(f"coe_R_1:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")

    print("利用Mag_quantile 和 Ang_exp 进行计算 ")
    coe_R_2 = np.array(Mag_quantile) - np.array(Ang_exp)
    auroc, fpr95, aupr, _, _, _ = selfeval.self_eval(coe_R_2, binary_list)
    print(f"coe_R_2:    {'auroc'.rjust(30)}: {auroc:.2f}{'fpr95'.rjust(30)}: {fpr95:.2f}{'aupr'.rjust(30)}: {aupr:.2f}")
