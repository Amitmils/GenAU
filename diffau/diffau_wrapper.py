from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from typing import Optional,List
import sounddevice as sd
import matplotlib.pyplot as plt
from torchaudio import load
from data_module import AmbisonicSpecDataModule
from model import ScoreModel
from diffau.util.other import stft_si_sdr,stft_sdr,si_sdr_batch,sdr_batch,stft_sdr_per_band
from diffau.util.checkpoint_loader import safe_load_from_checkpoint
import torch.nn.functional as F


# Class for Inference on trained Diffusion Models
class DiffAU:
    def __init__(self, step1_model_path : Optional[str], data_path : Optional[str], step2_model_path : Optional[str] = None ,split : str = 'test', N : int = 30):
        self.models : List[ScoreModel] = list()
        if not(os.path.exists(data_path)):
            raise ValueError(f"Data path {data_path} does not exist.")
        self.data_path = data_path

        if os.path.exists(step1_model_path):
            self.models.append(safe_load_from_checkpoint(ScoreModel, step1_model_path, base_dir=self.data_path))
            self.models[0].setup() # setup the datamodule
            self.datamodule = self.models[0].data_module
        else:
            raise ValueError(f"Step 1 model path {step1_model_path} does not exist.")
        
        if step2_model_path is not None and step1_model_path!=step2_model_path:
            if os.path.exists(step2_model_path):
                self.models.append(safe_load_from_checkpoint(ScoreModel, step2_model_path, base_dir=data_path))
            else:
                raise ValueError(f"Step 2 model path {step2_model_path} does not exist.")
        else:
            print("No step 2 model provided. Only step 1 model will be used for inference.")
        
        self.data_df = self.datamodule.meta_data.copy().query("type == @split") #this will be the data that will be used for inference
        self.N = N # number of iterations in the diffusion process
        self.input_channels = (self.models[0].input_order+1)**2
        self.output_channels = (self.models[-1].output_order+1)**2
        self.samples_in_segment = int((self.datamodule.num_frames - 1) * self.datamodule.num_frames//2) #We assume spectogram is num_frames**2 

    def stft(self,x):
        return self.models[0]._stft(x)

    def run_data_df(self,audio_id_start : int = 0, audio_id_end : int = 10000, df : Optional[pd.DataFrame] = None, verbose : bool = True):
        if df is None:
            #take loaded df
            df_to_run = self.data_df.iloc[audio_id_start:audio_id_end]
        else:
            df_to_run = df
        num_audios = len(df_to_run)
        self.stft_sdr_channel_cols = [f'stft_sdr_ch_{ch}' for ch in range(self.input_channels,self.output_channels)]


        gt_list = list()
        upscaled_list = list()
        print(f"Running {num_audios} audios.")
        for i,row in tqdm(df_to_run.iterrows(), total=num_audios):
            x,sr = load(os.path.join(self.data_path,row['type'],f"{row['type']}_{i}.wav")) # TODO add audio file path to meta data df on creations - this is a WA
            x_hat , metrics = self(x)
            self.data_df.loc[i,'full_audio_stft_sdr'] = metrics['audio_stft_sdr'].item()
            self.data_df.loc[i,self.stft_sdr_channel_cols] = metrics['stft_sdr'].cpu().numpy().tolist()
            upscaled_list.append(x_hat.cpu())
            gt_list.append(x.cpu())
        if verbose:
            self.print_metrics_analysis()

        return torch.stack(gt_list,dim=0),torch.stack(upscaled_list,dim=0)

    def print_metrics_analysis(self,metric : List[str] = ['stft_sdr']):
        for m in metric:
            title = f"### {m.upper().replace('_',' ')} ###"
            print()
            print("#" * len(title))
            print(title)
            print("#" * len(title))
            print()

            relevant_cols = [col for col in self.data_df.columns if m in col]
            avg_ch_stft_sdr = list()
            std_stft_sdr = list()
            for col in relevant_cols:
                avg_ch_stft_sdr.append(self.data_df[col].mean())
                std_stft_sdr.append(self.data_df[col].std())
            
            print(f"Avg Over All Channels : {np.mean(avg_ch_stft_sdr):.2f} dB")
            print(f"STD Over All AVG Channels : {np.std(avg_ch_stft_sdr):.2f} dB")
            print(f"Max of All AVG Channels : {np.max(avg_ch_stft_sdr):.2f} dB")
            print(f"Min of All AVG Channels : {np.min(avg_ch_stft_sdr):.2f} dB")
            print()

            for i,col in enumerate(relevant_cols):
                print(f"Ch. {col.split('_')[-1]} : Avg {avg_ch_stft_sdr[i]:.2f} dB , STD {std_stft_sdr[i]:.2f} dB")

    def pad_and_chunk_audio(self, x : torch.Tensor):
        x = x[:self.output_channels] #truncate if more channels than final step output order
        x = F.pad(x,(0,self.samples_in_segment - x.shape[1] % self.samples_in_segment))
        assert x.shape[1] % self.samples_in_segment == 0, "Input audio length must be a multiple of samples_in_segment. (STFT needs to be num_framesxnum_frames long)"
        chunks = x.unfold(dimension=1,size=self.samples_in_segment,step=self.samples_in_segment).permute(1,0,2) # (num_chunks, num_channels, samples_in_segment)
        return x,chunks
        
    def __call__(self,x):
        x,chunks= self.pad_and_chunk_audio(x)
        x_hat = []
        #TODO make this more efficient by processing multiple chunks at once
        for curr_chunk in tqdm(chunks,leave=False, desc="Processing chunks"):
            y_chunk = curr_chunk.clone()[:self.input_channels]
            for model in self.models:
                with torch.no_grad():
                    gen_channels = model.upscale(y_chunk,N=self.N)
                y_chunk = torch.concat((y_chunk,gen_channels),dim=0)
            x_hat.append(y_chunk)
        x_hat = torch.cat(x_hat,dim=1)
        x_HOA = x[self.input_channels:]
        x_HOA_hat = x_hat[self.input_channels:]
        metrics = dict()
        metrics['stft_si_sdr'] = stft_si_sdr(self.stft(x_HOA), self.stft(x_HOA_hat))
        metrics['stft_sdr'],metrics['audio_stft_sdr'] = stft_sdr(self.stft(x_HOA), self.stft(x_HOA_hat),full_audio_sdr=True)
        metrics['energy_ber_bin'] = {"GT" : self.stft(x_HOA).pow(2).sum(-1).log10()*10, "EST" : self.stft(x_HOA_hat).pow(2).sum(-1).log10()*10}
        metrics['si_sdr'] = si_sdr_batch(x_HOA, x_HOA_hat)
        metrics['sdr'] = sdr_batch(x_HOA, x_HOA_hat)
        metrics['per_band_stft_sdr'] = stft_sdr_per_band(self.stft(x_HOA), self.stft(x_HOA_hat))
        return x_hat,metrics
