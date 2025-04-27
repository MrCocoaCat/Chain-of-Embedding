import os
import sys
import time
import random

import pickle
import argparse
import scipy.spatial
import math
import json
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize
import seaborn as sns
from collections import Counter
import torch.nn.functional as F
import numpy as np
import pickle
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

#project_root_path = os.environ["PROJECT_PATH"]
project_root_path = "C:/Users/liyubo/Documents/GitHub/Chain-of-Embedding/"
sys.path.append(project_root_path)
from Data.load_data import DatasetInfo
from prompt_pool import *
from score import OutputScoreInfo, CoEScoreInfo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Inference:
    def __init__(self, model_info: dict, dataset_info: dict, verbose: dict):
        self.model_info = model_info
        self.dataset_info = dataset_info
        self.verbose = verbose
        self.model = self.model_info["model_ckpt"]
        self.model_name = self.model_info["model_name"]
        self.config = self.model_info["model_config"]
        self.generation_config = self.model_info["generation_config"]
        self.tokenizer = self.model_info["tokenizer"]
        self.max_output_token = self.model_info["max_output_token"]
        
        self.dataset_name = self.dataset_info["dataset_name"]
        self.data_loader = DatasetInfo(self.dataset_name)
        self.data_all = self.data_loader.data
        self.data_size = self.data_loader.data_size
        self.language = self.dataset_info["language"]

        self.sample_info = {}


    def dataset_inference(self):
        self.greedy_inference()
        

    def greedy_inference(self):
        for i in tqdm(range(self.data_size)):
            print("*"*30 + f" index {str(i)} " + "*"*30)
            sample = self.data_all[i]
            input_data, output_data, model_input, input_ids = self.parse_input(sample)
            self.sample_info = {
                "input": {
                    "raw_input_data": input_data,
                    "model_input": model_input,
                    "model_input_ids": input_ids,
                },
                "output": {
                    "raw_output_data": output_data,
                }
            }

            with torch.no_grad():
                generation_output = self.model_inference()
                self.sample_info["output"]["output_scores"] = generation_output.scores
                self.sample_info["output"]["output_seq"] = generation_output.sequences
                self.sample_info["output"]["attentions"] = generation_output.attentions
                self.sample_info["output"]["all_token_hidden_states"] = generation_output.hidden_states # output_len x layer_num x sampling_num x beam_search x hidden_dim
                self.sample_info["output"]["output_len"] = min(self.max_output_token, len(generation_output.scores))

                output_seq, maxprob, ppl, entropy = self.print_output()
                output = {'id': i,
                        'answer_type': sample["answer_type"] if self.dataset_name == "theoremqa" else "",
                        'input_seq': self.sample_info["input"]["model_input"],
                        'output_seq': output_seq,
                        'maxprob': maxprob,
                        'ppl': ppl,
                        'entropy': entropy}
                if self.verbose["save_output"]: self.save_output(output, i)

                hidden_states = self.print_hidden_states()
                if self.verbose["save_hidden_states"]: self.save_hidden_states(hidden_states, i)

                CoE_score = self.print_CoE_score()
                if self.verbose["save_coe_score"]: self.save_CoE_score(CoE_score, i) 
                if self.verbose["save_coe_figure"]: self.save_CoE_figure(hidden_states, i)



    def model_inference(self):
        # 模型进行计算推理额时间
        input_ids = self.sample_info["input"]["model_input_ids"]
        self.model.eval()
        terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")] \
            if "Llama" in self.model_name else self.tokenizer.eos_token_id
    
        time_start = time.time()
        # 开始推理
        generation_output = self.model.generate(
            # 将输入的ID张量移动到指定的设备（如GPU）上
            input_ids=input_ids.to(device),
            # 设置填充标记的ID为分词器的结束标记ID
            pad_token_id=self.tokenizer.eos_token_id,
            # 设置结束标记的ID为 terminators 变量所指定的值
            eos_token_id=terminators,
            # 使用预定义的生成配置
            generation_config=self.generation_config,
            # 让生成方法返回一个字典，包含生成的结果和其他信息
            return_dict_in_generate=True,
            # 设置最大的新生成标记数量
            max_new_tokens=self.max_output_token,
            # 要求生成过程中输出注意力权重
            output_attentions=True,
            # 要求生成过程中输出隐藏层状态
            output_hidden_states=True,
            # 要求生成过程中输出每个时间步的分数
            output_scores=True,
            # 不使用采样策略，而是使用贪心搜索进行生成
            do_sample=False,
        )
        time_end = time.time()
        print(f'inference time: {round(time_end - time_start, 4)}')
        
        return generation_output


    def parse_input(self, sample):
        input_data = sample[self.language]
        output_data = sample["answer"]
        # 将数据集替换到提示工程中
        model_input = DATASET_PROMPTS[self.dataset_name].replace("{input_data}", input_data)
        if self.dataset_name == "theoremqa":
            model_input = model_input.replace("{answer_type}", sample["answer_type"])
        # 编码
        input_ids = self.tokenizer.apply_chat_template([{"role": "user", "content": model_input}], 
                            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_len = len(input_ids[0])
        print(f"********** Input Text (length: {input_len}) **********\n{input_data}\n")
        print(f"********** Input ID **********\n{input_ids}\n")
        return input_data, output_data, model_input, input_ids


    def print_output(self):
        output_scores = self.sample_info["output"]["output_scores"]
        output_seq = self.sample_info["output"]["output_seq"]
        true_output = self.sample_info["output"]["raw_output_data"]
        output_len = self.sample_info["output"]["output_len"]

        output_seq = self.tokenizer.decode(output_seq[0][-output_len:])
        print(f"********** Model-generated Text (length: {output_len}) **********\n{output_seq}\n")
        print(f"********** True Output Text **********\n{true_output}\n")

        outputinfo = OutputScoreInfo(output_scores)
        maxprob = outputinfo.compute_maxprob()
        ppl = outputinfo.compute_ppl()
        entropy = outputinfo.compute_entropy()
        print(f"********** Output Info: **********\nmaxprob {maxprob}; perplexity {ppl}; entropy {entropy}\n")

        return output_seq, maxprob, ppl, entropy

    
    def save_output(self, output, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Output', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(output, file)
    

    def print_hidden_states(self):
        hidden_states = self.sample_info["output"]["all_token_hidden_states"]  #所有的hidden层的tensor
        output_len = self.sample_info["output"]["output_len"]

        layer_num = len(hidden_states[1])
        hs_all_layer = []
        for j in range(layer_num):
            all_pos_hs = np.array([np.array(hidden_states[pos][j][0][0].cpu()) for pos in range(0, output_len)]) #
            hs_all_layer.append(np.mean(all_pos_hs, axis=0))
        hidden_states = hs_all_layer
        print(f"********** Hidden State Size: **********\n{np.array(hidden_states).shape}\n")

        return hidden_states


    def save_hidden_states(self, hidden_states, i):
        hs = {'hidden_states': hidden_states}
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/HiddenStates', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(hs, file)


    def print_CoE_score(self):
        hidden_states = self.sample_info["output"]["all_token_hidden_states"]
        output_len = self.sample_info["output"]["output_len"]
        layer_num = len(hidden_states[1])

        hs_all_layer = []
        for j in range(layer_num):
            all_pos_hs = np.array([np.array(hidden_states[pos][j][0][0].cpu()) for pos in range(0, output_len)])
            hs_all_layer.append(np.mean(all_pos_hs, axis=0))

        coescoreinfo = CoEScoreInfo(hs_all_layer)
        _, coe_mag, _ = coescoreinfo.compute_CoE_Mag()
        _, coe_ang, _ = coescoreinfo.compute_CoE_Ang()
        coe_r = coescoreinfo.compute_CoE_R()
        coe_c = coescoreinfo.compute_CoE_C()

        print(f"********** CoE Score Info: **********\nMag {coe_mag}; Ang {coe_ang}; R {coe_r}; C {coe_c}\n")
        return {
            "Mag": coe_mag,
            "Ang": coe_ang,
            "R": coe_r,
            "C": coe_c
        }


    def save_CoE_score(self, CoE_score, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/CoE', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(CoE_score, file)   


    def save_CoE_figure(self, hidden_states, i):
        embeddings = PCA(n_components=2, random_state=2024).fit_transform(np.array(hidden_states))

        fig = plt.figure(figsize=(14, 8))
        #fig.suptitle('Embedding Trajectory under Correct/Incorrect Samples', fontsize=40, fontweight='bold')
        ax1 = fig.add_subplot(1, 1, 1, facecolor='w')

        traj_x = np.array(embeddings[:, 0])
        traj_y = np.array(embeddings[:, 1])

        ax1.scatter(traj_x, traj_y, color='blue', alpha=1.0, edgecolor='white', s=200)
        ax1.plot(traj_x, traj_y, color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax1.text(0, 0, "Origin (0,0)", color='black', fontsize=10)

        ax1.set_xlabel('X-axis', fontsize=24, fontweight='bold')
        ax1.set_ylabel('Y-axis', fontsize=24, fontweight='bold')
        
        '''save'''
        filedir = os.path.join(project_root_path, f'Figure/{self.language}', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        plt.savefig(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.png'), bbox_inches='tight', pad_inches=0)


class InferenceFromOutput(Inference):
    def __init__(self, model_info, dataset_info, verbose):
        super().__init__(model_info, dataset_info, verbose)

    def get_output(self, i):
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Output', self.model_name,
                               self.dataset_name)
        # if not os.path.exists(filedir):
        #     os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(i) + '.pkl'), 'rb') as file:
            #pickle.dump(output, file)
            loaded_data = pickle.load(file)
        return  loaded_data

    def greedy_inference(self):
        for i in tqdm(range(self.data_size)):
            print("*"*30 + f" index {str(i)} " + "*"*30)
            sample = self.data_all[i]
            self.sample_info = self.get_sample_info(i)
            with torch.no_grad():
                # 将模型保存下来
                self.get_hidden_layer(i)
                # generation_output = self.model_inference()
                # 读取output
                output_seq, maxprob, ppl, entropy = self.print_output()
                output = {'id': i,
                        'answer_type': sample["answer_type"] if self.dataset_name == "theoremqa" else "",
                        'input_seq': self.sample_info["input"]["model_input"],
                        'output_seq': output_seq,
                        'maxprob': maxprob,
                        'ppl': ppl,
                        'entropy': entropy}
                if self.verbose["save_output"]: self.save_output(output, i)

                hidden_states = self.print_hidden_states()
                if self.verbose["save_hidden_states"]: self.save_hidden_states(hidden_states, i)
                CoE_score = self.print_CoE_score()
                if self.verbose["save_coe_score"]: self.save_CoE_score(CoE_score, i)
                if self.verbose["save_coe_figure"]: self.save_CoE_figure(hidden_states, i)

    def get_hidden_layer(self,id):
        """
        将模型的layer 保存下来，
        :return:
        """
        filedir = os.path.join(project_root_path, 'OutputInfo',self.language,'HiddenLayer', self.model_name,
                               self.dataset_name)

        file_path = os.path.join(filedir, self.dataset_name+"_"+str(id)+".pt")
        #torch.save(output_scores, file_path)
        print(f" 读取 {file_path} Tensor")
        loaded_tensor = torch.load(file_path)
        self.sample_info["output"]["output_scores"] = loaded_tensor
        #print(loaded_tensor)



    def get_sample_info(self,id):
        """
        保存sample_info
        :return:
        """
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Sample_info', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(id) + '.pkl'), 'rb') as file:
            sample_info = pickle.load(file)
        return  sample_info

class InferenceSaveLayer(Inference):
    def __init__(self, model_info, dataset_info, verbose):
        super().__init__(model_info, dataset_info, verbose)

    def greedy_inference(self):
        r = range(self.data_size)
        r = range(100)
        for i in tqdm(r):
            print("*"*30 + f" index {str(i)} " + "*"*30)
            sample = self.data_all[i]
            input_data, output_data, model_input, input_ids = self.parse_input(sample)
            self.sample_info = {
                "input": {
                    "raw_input_data": input_data,
                    "model_input": model_input,
                    "model_input_ids": input_ids,
                },
                "output": {
                    "raw_output_data": output_data,
                }
            }

            with torch.no_grad():
                generation_output = self.model_inference()
                self.sample_info["output"]["output_scores"] = generation_output.scores
                self.sample_info["output"]["output_seq"] = generation_output.sequences
                self.sample_info["output"]["attentions"] = generation_output.attentions
                self.sample_info["output"]["all_token_hidden_states"] = generation_output.hidden_states # output_len x layer_num x sampling_num x beam_search x hidden_dim
                self.sample_info["output"]["output_len"] = min(self.max_output_token, len(generation_output.scores))
                # 将模型保存下来
                self.save_hidden_layer(i)

                output_seq, maxprob, ppl, entropy = self.print_output()
                output = {'id': i,
                        'answer_type': sample["answer_type"] if self.dataset_name == "theoremqa" else "",
                        'input_seq': self.sample_info["input"]["model_input"],
                        'output_seq': output_seq,
                        'maxprob': maxprob,
                        'ppl': ppl,
                        'entropy': entropy}
                if self.verbose["save_output"]:
                    self.save_output(output, i)
                # 将sample_info 保存

                self.save_sample_info(self.sample_info,i)
                hidden_states = self.print_hidden_states()
                if self.verbose["save_hidden_states"]: self.save_hidden_states(hidden_states, i)

                CoE_score = self.print_CoE_score()
                if self.verbose["save_coe_score"]: self.save_CoE_score(CoE_score, i)
                if self.verbose["save_coe_figure"]: self.save_CoE_figure(hidden_states, i)

    def save_hidden_layer(self,id):
        """
        将模型的layer 保存下来，
        :return:
        """
        filedir = os.path.join(project_root_path, 'OutputInfo',self.language,'HiddenLayer', self.model_name,
                               self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        output_scores = self.sample_info["output"]["output_scores"]
        file_path = os.path.join(filedir, self.dataset_name+"_"+str(id)+".pt")
        torch.save(output_scores, file_path)
        print(f"Tensor 已保存到 {file_path}")

    def save_sample_info(self,sample_info,id):
        """
        保存sample_info
        :return:
        """
        filedir = os.path.join(project_root_path, f'OutputInfo/{self.language}/Sample_info', self.model_name, self.dataset_name)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with open(os.path.join(filedir, self.dataset_name + '_' + str(id) + '.pkl'), 'wb') as file:
            pickle.dump(sample_info, file)