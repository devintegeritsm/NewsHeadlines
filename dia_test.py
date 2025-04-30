import soundfile as sf
import torch
from dia.model import Dia


if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
try:
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
    if device.type == "cpu":
        model.model.to(torch.float32)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

output = model.generate(text, use_torch_compile=False, verbose=True)

# sf.write("dia_test.mp3", output, 44100)
sf.write("dia_test.wav", output, 44100)