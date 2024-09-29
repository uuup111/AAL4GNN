import argparse
import os
import threading
from itertools import product

from task_config import OGBN_ARXIV, HYPER_CONFIG_FULL


# copy from the AAL
class RunThreadClass(threading.Thread):
    def __init__(self, command_list):
        self.stdout = None
        self.stderr = None
        self.command_list = command_list
        threading.Thread.__init__(self)

    def run(self):
        for cmd in self.command_list:
            os.system(cmd)


def hyperParam_option(parser):
    parser.add_argument('-task', type=str, default='ogbn_arxiv')
    parser.add_argument('-output_dir', type=str, default='./output')
    parser.add_argument('-hyper_config', type=str, default='hyper_config_full')


def hyperConfigParser_to_str(config_name):
    if config_name == 'hyper_config_full':
        return HYPER_CONFIG_FULL


def get_task_info(args):
    if args.task == 'ogbn_arxiv':
        return OGBN_ARXIV


def get_all_hyperConfigs(config_dict):
    all_configs = list(product(*list(config_dict.values())))
    all_keys = list(config_dict.keys())
    all_hyperConfigs = [{k: v for k, v in zip(all_keys, config)} for config in all_configs]
    return all_hyperConfigs


def get_run_str(args, config_, task_info):
    # TODO finish the cmd to call the GNN run
    # 将config和task_info中的参数拼接到cmd运行指令中去，调用run.py完成GNN任务的运行
    run_command = 'python run.py'
    output_dir = ''
    return run_command, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add some hyperParam to parser
    hyperParam_option(parser)
    args = parser.parse_args()
    # get some task infos
    task_info = get_task_info(args)
    # get all the hyperConfigs space
    hyperConfig_str = hyperConfigParser_to_str(args.hyper_config)
    all_hyperConfigs = get_all_hyperConfigs(hyperConfig_str)

    all_threads = []
    run_commands = []
    all_results = {}
    # loop from the all_hyperConfigs with multiple threads
    for config_ in all_hyperConfigs:
        run_command, output_dir = get_run_str(args, config_, task_info)
        run_commands.append(run_command)
    # just one thread now, leaving for the multi threads in the future
    this_thread = RunThreadClass(run_commands)
    all_threads.append(this_thread)
    this_thread.start()
    for thread in all_threads:
        thread.join()

    # output the result


