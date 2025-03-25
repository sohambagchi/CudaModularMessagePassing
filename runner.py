import os
import sys
from pprint import pprint
import subprocess
import time

import pandas as pd

output_folder = 'output'

X_SCOPE = list()
Y_SCOPE = list()
PRODUCER_FENCE = list()
CONSUMER_FENCE = list()
MEMORY_ORDER = list()
GPU_P_GPU_C = list()
GPU_P_CPU_C = list()
CPU_P_GPU_C = list()
CPU_P_CPU_C = list()
WEAK_BEHAVIORS = list()

global_filename = sys.argv[1] if len(sys.argv) > 1 else 'message_passing.csv'
# message_passing_data_frame = pd.DataFrame(columns=['x_scope', 'y_scope', 'producer_fence', 'consumer_fence', 'memory_order', 'gpu_p_gpu_c', 'gpu_p_cpu_c', 'cpu_p_gpu_c', 'cpu_p_cpu_c', 'weak_behaviors'])


def does_dataitem_exist(data_frame, data_item):
    for index, row in data_frame.iterrows():
        if row['x_scope'] == data_item['x_scope'] and row['y_scope'] == data_item['y_scope'] and row['producer_fence'] == data_item['producer_fence'] and row['consumer_fence'] == data_item['consumer_fence'] and row['memory_order'] == data_item['memory_order']:
            if pd.isna(row['gpu_p_gpu_c']) or pd.isna(row['gpu_p_cpu_c']) or pd.isna(row['cpu_p_gpu_c']) or pd.isna(row['cpu_p_cpu_c']):
                # print(row)
                return False
            return True
    return False

def create_dataitem(x_scope, y_scope, producer_fence_order, producer_fence_scope, consumer_fence_order, consumer_fence_scope, memory_order, gpu_p_gpu_c=None, gpu_p_cpu_c=None, cpu_p_gpu_c=None, cpu_p_cpu_c=None, weak_behaviors=None):
    
    if producer_fence_order == 'PRODUCER_NO_FENCE':
        producer_fence_scope = '-'
        
    if consumer_fence_order == 'CONSUMER_NO_FENCE':
        consumer_fence_scope = '-'
    
    return {
        'x_scope': x_scope,
        'y_scope': y_scope,
        'producer_fence': f"{producer_fence_order.replace('PRODUCER_FENCE_', '')} {producer_fence_scope.replace('PRODUCER_FENCE_SCOPE_', '')}",
        'consumer_fence': f"{consumer_fence_order.replace('CONSUMER_FENCE_', '')} {consumer_fence_scope.replace('CONSUMER_FENCE_SCOPE_', '')}",
        'memory_order': memory_order,
        'gpu_p_gpu_c': gpu_p_gpu_c,
        'gpu_p_cpu_c': gpu_p_cpu_c,
        'cpu_p_gpu_c': cpu_p_gpu_c,
        'cpu_p_cpu_c': cpu_p_cpu_c,
        'weak_behaviors': weak_behaviors
    }
    

def append_to_dataframe(df, data):
    existing_index = None
    for index, row in df.iterrows():
        if row['x_scope'] == data['x_scope'] and row['y_scope'] == data['y_scope'] and row['producer_fence'] == data['producer_fence'] and row['consumer_fence'] == data['consumer_fence'] and row['memory_order'] == data['memory_order']:
            existing_index = index
            break

    if existing_index is not None:
        if pd.isna(df.at[existing_index, 'gpu_p_gpu_c']) or pd.isna(df.at[existing_index, 'gpu_p_cpu_c']) or pd.isna(df.at[existing_index, 'cpu_p_gpu_c']) or pd.isna(df.at[existing_index, 'cpu_p_cpu_c']):
            df.at[existing_index, 'gpu_p_gpu_c'] = data['gpu_p_gpu_c']
            df.at[existing_index, 'gpu_p_cpu_c'] = data['gpu_p_cpu_c']
            df.at[existing_index, 'cpu_p_gpu_c'] = data['cpu_p_gpu_c']
            df.at[existing_index, 'cpu_p_cpu_c'] = data['cpu_p_cpu_c']
            df.at[existing_index, 'weak_behaviors'] = data['weak_behaviors']
        else:
            new_row = pd.Series(data)
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    else:
        new_row = pd.Series(data)
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    return df

def export_dataframe(df, filename=global_filename):
    df.to_csv(filename, index=False)

def import_dataframe(filename=global_filename):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=['x_scope', 'y_scope', 'producer_fence', 'consumer_fence', 'memory_order', 'gpu_p_gpu_c', 'gpu_p_cpu_c', 'cpu_p_gpu_c', 'cpu_p_cpu_c', 'weak_behaviors'])

message_passing_data_frame = import_dataframe()

for file in os.listdir(output_folder):
    # print(file)
    if 'RLX_RLX' not in file:
        continue
    
    if 'PRODUCER_NO_FENCE' not in file or 'CONSUMER_NO_FENCE' not in file:
        continue
    
    if 'SYS' in file:
        # print(file)
        continue
    
    # if ('REL_ACQ' in file or 'REL_RLX' in file or 'RLX_ACQ' in file) and ('PRODUCER_NO_FENCE' not in file or 'CONSUMER_NO_FENCE' not in file):
    #     # print(file)
    #     continue
    
    # if any(x in file.lower() for x in ['rel_acq', 'rel_rlx', 'rlx_acq']):
    #     if 'producer_no_fence' not in file.lower() or 'consumer_no_fence' not in file.lower():
    #         continue
    
    
    file_parameters = file.replace('.out', '').split('-')[1:]
    
    if ('ACQ_REL' in file_parameters[3] and 'SC' in file_parameters[5]) or ('ACQ_REL' in file_parameters[5] and 'SC' in file_parameters[3]):
        continue
    
    if ('CTA' in file_parameters[1] and 'CTA' not in file_parameters[0]):
        continue
    
    data_item = create_dataitem(file_parameters[0], file_parameters[1], file_parameters[2], file_parameters[3], file_parameters[4], file_parameters[5], file_parameters[6])
    
    # pprint(data_item)
    
    # if not does_dataitem_exist(message_passing_data_frame, data_item):
        # message_passing_data_frame = append_to_dataframe(message_passing_data_frame, data_item)
        
    print(file)
    start_time = time.time()
    
    
    result = subprocess.Popen(['./output/' + file, '-i', '1000000', '-t', '20', '-s', '20000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    while result.poll() is None:
        time.sleep(10)
        elapsed_time = time.time() - start_time
        if elapsed_time > 100:
            result.terminate()
            print(f'Terminated {file}')
            break
    
    stdout, stderr = result.communicate()

    wb = 0
    
    for line in stdout.decode().split('\n'):
        if 'GPU-P GPU-C' in line:
            data_item['gpu_p_gpu_c'] = line.split('-')[-1]
            wb = wb + int(line.split('-')[-1].split('|')[0].split(':')[-1].strip())
        elif 'GPU-P CPU-C' in line:
            data_item['gpu_p_cpu_c'] = line.split('-')[-1]
            wb = wb + int(line.split('-')[-1].split('|')[0].split(':')[-1].strip())
        elif 'CPU-P GPU-C' in line:
            data_item['cpu_p_gpu_c'] = line.split('-')[-1]
            wb = wb + int(line.split('-')[-1].split('|')[0].split(':')[-1].strip())
        elif 'CPU-P CPU-C' in line:
            data_item['cpu_p_cpu_c'] = line.split('-')[-1]
            wb = wb + int(line.split('-')[-1].split('|')[0].split(':')[-1].strip())
            
    data_item['weak_behaviors'] = wb
    
    # pprint(data_item)
    
    message_passing_data_frame = append_to_dataframe(message_passing_data_frame, data_item)
    
    export_dataframe(message_passing_data_frame)
    
    
    
    # print(file_parameters)
    # exit()