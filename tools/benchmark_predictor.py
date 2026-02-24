import time
import torch
import numpy as np
from mvector.swallow_predictor import SwallowPredictor
from yeaudio.audio import AudioSegment

def benchmark():
    # Load audio-like data
    duration = 5.0 # 5 seconds
    sample_rate = 16000
    samples = np.random.randn(int(duration * sample_rate)).astype(np.float32)
    audio = AudioSegment(samples=samples, sample_rate=sample_rate)
    
    print("Initializing SwallowPredictor...")
    predictor = SwallowPredictor(use_gpu=torch.cuda.is_available())
    
    # Warmup (First run involves model loading and potential torch.compile overhead)
    print("Warming up (this may take a while if torch.compile is active)...")
    t0 = time.time()
    predictor.analyze(audio, "这是一段测试文本")
    print(f"Warmup done in {time.time() - t0:.4f}s")
    
    print("Benchmarking subsequent runs...")
    start_time = time.time()
    num_runs = 5
    for i in range(num_runs):
        t1 = time.time()
        predictor.analyze(audio, "这是一段二合一吞音检测测试文本，用于性能评估")
        print(f"Run {i+1}: {time.time() - t1:.4f}s")
    
    avg_time = (time.time() - start_time) / num_runs
    print(f"\nAverage analysis time: {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark()
