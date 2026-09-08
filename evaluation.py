import numpy as np
import torchaudio.functional as F
from visqol import VisqolApi
import torch
from torchmetrics.audio import PerceptualEvaluationSpeechQuality

class Evaluator: 
    def __init__(self):
        self.target_sr = 16000 
        mode = "speech" 
        self.visqol_api = VisqolApi()
        self.visqol_api.create(mode=mode)
        self.pesq_api = PerceptualEvaluationSpeechQuality(self.target_sr , 'wb')
    
    def match_length(self, ref, sr_pred):
        # used in case the lenghts of the degraded and source are not of same length
        min_len = min(len(ref), len(sr_pred))
        return ref[:min_len], sr_pred[:min_len]

    def sample_to_correct_rate(self, hr_audio, sr_audio, current_sr):
        if current_sr != self.target_sr:
            # we resample to 16k htz:
            hr_target = F.resample(hr_audio, orig_freq=current_sr, new_freq=self.target_sr)
            sr_target = F.resample(sr_audio, orig_freq=current_sr, new_freq=self.target_sr)
        else:
            hr_target = hr_audio
            sr_target = sr_audio
        return hr_target, sr_target

    def evaluate_pesq(self, hr_audio, sr_audio, current_sr):
        score = []
        for hr_target, sr_target  in zip(hr_audio, sr_audio):
            hr_target , sr_target = self.sample_to_correct_rate(hr_target, sr_target, current_sr)
            hr_target, sr_target = self.match_length(hr_target, sr_target)
            score.append(self.pesq_api(sr_target.cpu(), hr_target.cpu()))
        return sum(score)/len(score)

    def evaluate_visqol(self, hr_audio, sr_audio, current_sr):
        """
        this function takes in a single batch of audio and returns the average computed across the batch
        hr_audio: torch_tensor [B, time](can exist on gpu)-> high resolution sample(single sample)
        sr_audio: torch_tensor -> super resolution sample
        current_sr : > the resolution of the hr_audio and sr_audio that we feed

        note: the evaluation metric works only best at 16k htz, which means that
        we need to resample to 16k htz if current_sr != 16k htz
        """
        visqol = []
        for hr_target , sr_target in zip(hr_audio, sr_audio):
            hr_target, sr_target = self.sample_to_correct_rate(hr_target, sr_target, current_sr)

            hr_target, sr_target = self.match_length(hr_target, sr_target)

            hr_target = hr_target.detach().cpu().numpy().astype(np.float64)
            sr_target = sr_target.detach().cpu().numpy().astype(np.float64)
            
            try:
                similarity_result = self.visqol_api.measure_from_arrays(hr_target, sr_target, sample_rate=self.target_sr)
                visqol.append(similarity_result.moslqo)
            except Exception as e:
                print("error in measuring visqol")
                visqol.append(float('nan'))
        return sum(visqol)/ len(visqol)
    


if __name__ == "__main__":
    # Example usage directly from a PyTorch training/eval loop
    evaluator = Evaluator() 
    
    # Simulating the ground truth and model output tensors (e.g., at 8kHz)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_hr = []
    dummy_sr = []
    for i in range(1, 4):
        dummy_hr_target = torch.randn(80000, device=device) # 10 seconds of 8kHz audio
        dummy_sr_pred = torch.randn(8000, device=device)  # Slightly longer due to padding
        dummy_hr.append(dummy_hr_target)
        dummy_sr.append(dummy_sr_pred)

    
    # Pass the tensors directly
    score = evaluator.evaluate_visqol(dummy_hr, dummy_sr, current_sr=8000)
    
    if score is not None:
        print(f"ViSQOL MOS-LQO Score: {score:.3f}")

    dummy_hr_batch = torch.stack(dummy_hr)
    dummy_sr_batch = torch.stack(dummy_sr)
    pesq_score = evaluator.evaluate_pesq(dummy_hr_batch, dummy_sr_batch, current_sr = 8000)
    print(f"the pesq score is :{pesq_score}")