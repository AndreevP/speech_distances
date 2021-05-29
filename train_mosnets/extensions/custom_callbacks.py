import os
from pytorch_lightning.callbacks import Callback
import jsonlines
import sys
from subprocess import Popen, PIPE
import zipfile
import json
import fnmatch
import torch
import torch.distributed as dist

DEFAULT_IGNORE_LIST = [
     '*.zip*',
     '*.pth*',
     '*__pycache__*',
     '*.ipynb_checkpoints*',
     '*.jpg',
     '*.jpeg',
     '*.png',
     '*.wav',
     '*.mp4',
     '*.bmp',
     '*.mov',
     '*.mp3',
     '*.csv',
     '*.txt',
     '*.json',
     '*.tar.gz',
     '*.zip',
     '*.gzip',
     '*.7z',
     '*.ipynb',
     '*.coredump',
     'logs/*',
     'runs/*',
     'core.*',
     'lightning_logs/*',
     'test_exp/*',
     'experiments/*',
     'experiments_tb/*'
]


class run_validation_on_start(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        return trainer.run_evaluation()
    
class MetricLogger(Callback):
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.file_path = os.path.join(self.dirpath, 'output.jsonl')
    
    def on_validation_epoch_end(self, trainer, *args, **kwargs):
        if dist.is_initialized() and (dist.get_rank() > 0):
            return
        with jsonlines.open(self.file_path, mode='a') as writer:
            to_log = {}
            for key in trainer.logged_metrics:
                val = trainer.logged_metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                to_log[key] = val
            writer.write(to_log)
    """
    def on_train_epoch_end(self, trainer, *args,**kwargs):
        with jsonlines.open(self.file_path, mode='a') as writer:
#             print(trainer.logged_metrics)
            to_log = {}
            for key in trainer.logged_metrics:
                val = trainer.logged_metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                to_log[key] = val
            writer.write(to_log)
    """     

def collect_snapshot_list(command_root_dir, ignore_list=DEFAULT_IGNORE_LIST, full_path=False):
    results = []
    for file in os.listdir(command_root_dir):
        if check_list(file, ignore_list) and file[0] != '.':
            if os.path.isdir(os.path.join(command_root_dir, file)):
                for root, _, files in os.walk(os.path.join(command_root_dir, file)):
                    for sub_file in files:
                        if check_list(os.path.relpath(os.path.join(root, sub_file), command_root_dir), ignore_list):
                            results.append(os.path.join(root, sub_file))
            else:
                results.append(os.path.join(command_root_dir, file))

    if full_path:
        return results
    else:
        return [os.path.relpath(f, command_root_dir) for f in results]
    
    
def check_list(path, masks):
    for mask in masks:
        if fnmatch.fnmatch(path, mask):
            return False
    return True
    

class CodeSnapshotter(Callback):
    def __init__(self, folder, ignore_list=DEFAULT_IGNORE_LIST):
        self.root_path = folder
        self.ignore_list = ignore_list
        
    def on_train_start(self, trainer, pl_module):
        if dist.is_initialized() and (dist.get_rank() > 0):
            return
        if not os.path.isdir(self.root_path):
            os.makedirs(self.root_path)
        with open(os.path.join(self.root_path, 'bash_command.txt'), 'w+') as fout:
            fout.write(' '.join(sys.argv))
            
        command_root_dir = os.getcwd()
        with zipfile.ZipFile(os.path.join(self.root_path, 'snapshot.zip'), 'w') as snapshot:
            snapshot_list = collect_snapshot_list(command_root_dir, self.ignore_list)
            for file in snapshot_list:
                snapshot.write(file)
        print('Made snapshot of size {:.2f} MB'.format(
            os.path.getsize(os.path.join(self.root_path, 'snapshot.zip')) / (1024 * 1024)))
        

def get_process_output(command):
    if not isinstance(command, (list, tuple)):
        command = command.split(' ')
    process = Popen(command, stdout=PIPE, shell=True)
    output, err = process.communicate()
    exit_code = process.wait()
    return exit_code, output.decode()


class EnvironmentCollector(Callback):
    def __init__(self, folder):
        self.root_path = folder
        
    def on_train_start(self, trainer, pl_module):
        if dist.is_initialized() and (dist.get_rank() > 0):
            return
        is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
        if is_conda:
            print('Collecting environment using conda...', end=' ')
            code, res = get_process_output('conda env export')
        else:
            print('Collecting environment using pip...', end=' ')
            code, res = get_process_output('pip list')

        print('FAILED' if code != 0 else 'OK')
        with open(os.path.join(self.root_path, 'environment.txt'), 'w') as f:
            f.write(res)

            
class ParamsLogger(Callback):
    def __init__(self, folder):
        self.root_path = folder
        
    def on_train_start(self, trainer, pl_module):
        if dist.is_initialized() and (dist.get_rank() > 0):
            return
        with open(os.path.join(self.root_path, 'hparams.json'), 'w') as f:
            f.write(json.dumps(pl_module.hparams))
            
            
class VisualizationLogger(Callback):
    def __init__(self, folder, method, n_draw=10):
        self.root_path = folder
        self.method = method
        self.to_draw = n_draw
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        print('In visualization logger')
        
        ids = batch['id']
        for i in range(len(ids)):
            index = ids[i]
            if index < self.to_draw:
                one_input = {}
                for key in batch:
                    one_input[key] = batch[key][i]
                one_output = {}
                for key in outputs:
                    one_outputs[key] = outputs[key][i]
                figures = draw_results(one_input, one_output)
