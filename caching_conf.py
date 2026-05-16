"""Utility functions for T-Cache."""

import os
from pathlib import Path

def read_wav_text_files(filename, dataset_path):
    """Read prompt/target pairs and keep only entries with existing prompt WAV files."""
    prompt_wavs = []
    prompt_texts = []
    target_texts = []
    target_wavs = []

    with open(filename) as f:
        for line in f:
            target_wav, prompt_text, prompt_wav, target_text = (
                line.strip().split("|")
            )

            target_wavs.append(target_wav.strip())
            prompt_texts.append(prompt_text.strip())
            prompt_wavs.append(prompt_wav.strip())
            target_texts.append(target_text.strip())

    assert (
        len(target_wavs)
        == len(prompt_texts)
        == len(prompt_wavs)
        == len(target_texts)
    ), "The lengths should be equal"

    # Add the dataset path to prompt WAV files.
    full_prompt_wavs = [
        os.path.join(dataset_path, file_path)
        for file_path in prompt_wavs
    ]

    # Keep only valid files
    filtered_target_wavs = []
    filtered_prompt_texts = []
    filtered_prompt_wavs = []
    filtered_target_texts = []

    removed_files = []

    for target_wav, prompt_text, prompt_wav, target_text, full_path in zip(
        target_wavs,
        prompt_texts,
        prompt_wavs,
        target_texts,
        full_prompt_wavs,
    ):
        if Path(full_path).exists():
            filtered_target_wavs.append(target_wav)
            filtered_prompt_texts.append(prompt_text)
            filtered_prompt_wavs.append(full_path)
            filtered_target_texts.append(target_text)
        else:
            removed_files.append(full_path)

    # Report removed files
    if removed_files:
        print(f"Removed {len(removed_files)} missing file(s):")
        for file in removed_files:
            print(file)

    return (
        filtered_target_wavs,
        filtered_prompt_texts,
        filtered_prompt_wavs,
        filtered_target_texts,
    )

  

def cache_init(model_kwargs=None, num_steps=None):          
    """Create empty cache containers for each transformer layer."""
    cache_dic = {}
    cache = {}
    cache[-1]={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    

    for j in range(16):  # Number of transformer layers.
        cache[-1][j] = {}
        cache_dic['attn_map'][-1][j] = {}
    
    for i in range(num_steps):
        cache[i]={}
        for j in range(16):
            cache[i][j] = {}
    
    cache_dic['cache']  = cache
    current = {}
    current['num_steps'] = num_steps
    return cache_dic, current


def initialize_caching_mode(kwargs, num_steps=None, prompt_len=None,phone_len=None,stage_name=None,mask_layer=None):
    """Initialize baseline or T-Cache state for T2S and S2A inference stages."""
    assert num_steps is not None, "the number of steps should be provided"
    assert prompt_len is not None, "the prompt_len should be provided"
    
    cache_dic, current = cache_init(model_kwargs=kwargs, num_steps=num_steps)
    cache_dic_cfg, current_cfg = cache_init(model_kwargs=kwargs, num_steps=num_steps)
    

    current['prompt_cache']=kwargs.prompt_cache
    current_cfg['prompt_cache']=kwargs.prompt_cache
    # Store prompt and phone lengths.
    current['prompt_len']= prompt_len
    current['phone_len'] = phone_len

       
    if kwargs.mode=="baseline":
        current['use_baseline'] = True        
        current['normal_cfg'] = True          
        current_cfg['use_baseline'] = True        
        
    elif kwargs.mode=="tcache":
        current['use_tcache'] = True        
        current['normal_cfg'] = False     
        current['thres_step']= 4 
        current['prompt_prefix_len']= (current['phone_len'] + current['prompt_len']) if current['phone_len'] is not None else current['prompt_len']      
        
        current_cfg['use_tcache'] = True 
        current_cfg['thres_step']= 4 
    
   
    else:
        raise ValueError("The caching mode is not specified")

    # Nonlinear scheduling.
    
    if kwargs.mode=="tcache":
        if stage_name=="t2s":    

            # Nonlinear scheduling.
            N = 10
            T = num_steps

            assert N<T, "N should be less than T"  # Ensure T is greater than N.

            S = [round((T/((N-1)**(1.5)))*(k**1.5)) for k in range(N)] 
            S[-1]= num_steps-1 # Ensure the last scheduled step is the final inference step.

            current['cache_steps']= S
            current_cfg['cache_steps']= S
            current_cfg['stage_name']="t2s"
            current['stage_name']="t2s"     

        if stage_name=="s2a":
            if mask_layer==0:
                
                N = 8
                T = num_steps
                assert N<T, "N should be less than T"  # Ensure T is greater than N.
                S = [round((T/((N-1)**(1.5)))*(k**1.5)) for k in range(N)] 
                S[-1]= num_steps-1  
                
                current['cache_steps']= S  # Use nonlinear scheduling for the first codebook.
                current_cfg['cache_steps']= S
            
            # Use linear scheduling for the second codebook.
            elif mask_layer==1:
                current['cache_steps']= [0,3,6,9] 
                current_cfg['cache_steps']= [0,3,6,9]

            # Other codebooks have only one timestep, so feature caching is not needed.
            else:
                current['cache_steps']= [0,] 
                current_cfg['cache_steps']= [0,]

    return cache_dic, current , cache_dic_cfg , current_cfg
