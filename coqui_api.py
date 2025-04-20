import warnings
import numpy as np
import torch

# Filter out specific PyTorch warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Tell torch.load that numpy.core.multiarray.scalar is safe to unpickle
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
torch.serialization.safe_globals([np.core.multiarray.scalar])

_orig_load = torch.load
def _load_force_full(*args, **kwargs):
    # inject weights_only=False if not already specified
    if 'weights_only' not in kwargs:
        print("weights_only not in kwargs")
        kwargs['weights_only'] = False
    else:
        print("weights_only in kwargs")
        kwargs['weights_only'] = False

    return _orig_load(*args, **kwargs)

torch.load = _load_force_full

from TTS.api import TTS
import soundfile as sf
import time
import os



TMP_FOLDER = 'tmp'
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)


# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

# model_name = "tts_models/en/ljspeech/tacotron2-DDC"
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
# model_name = "tts_models/multilingual/multi-dataset/bark"
# model_name = "tts_models/multilingual/multi-dataset/your_tts"

print(model_name)

# Init TTS
# tts = TTS(model_name=model_name, progress_bar=True, gpu=False)
tts = TTS(model_name=model_name, progress_bar=True).to(device)
# tts = TTS(model_name).to(device)

# List speakers
print(tts.speakers)

def generate_audio(text: str, output_filename: str):
    tts.tts_to_file(text=text, 
                    # language="ru", 
                    # speaker="Craig Gutsy",
                    # speaker="Lidiya Szekeres",
                    file_path=output_filename)
    print(f"Audio file generated successfully: {output_filename}")
    return output_filename

if __name__ == "__main__":
    print("Starting tts test...")
    text1 = "Hello, how are you doing?"
    text = """DOGE Cuts Continue to Disrupt HHS; Supreme Court Backs Admin on Probationary Firings

Fallout continues from the sweeping personnel cuts across federal agencies directed by Elon Musk's Department of Government Efficiency (DOGE).

HHS Chaos: Mass firings at the Department of Health and Human Services (HHS), including the CDC, FDA, and NIH, on April 1st have caused significant disruption. Thousands lost their jobs, impacting critical functions like the CDC's Division of Violence Prevention, FOIA processing, antibiotic resistance labs, and the Substance Abuse and Mental Health Services Administration (SAMHSA, losing over 10% staff). HHS Secretary Robert F. Kennedy Jr. acknowledged some cuts might be "mistakes" and moved to reinstate some staff, but critics warn of severe risks to public health infrastructure and research capacity. The 988 National Suicide Prevention Lifeline is also reported to be facing understaffing issues exacerbated by the cuts.
Probationary Firings Upheld (Temporarily): The Supreme Court sided with the Trump administration, halting a lower court order that would have reinstated ~16,000 probationary federal employees fired as part of DOGE's downsizing. The 5-4 decision (Sotomayor, Jackson dissenting) was based on the finding that the non-profit organizations challenging the firings lacked legal standing. This allows the terminations to proceed for now, though a separate challenge by 19 states and D.C. in Maryland continues, potentially limiting the ruling's immediate nationwide impact. The administration cited poor performance for the firings, while challengers claimed the employees had positive reviews.

Analytical Take: The DOGE cuts are causing tangible disruption, particularly in public health, raising questions about the strategic coherence and potential long-term damage of Musk's efficiency drive. Secretary Kennedy Jr.'s attempt to walk back some HHS firings suggests internal recognition of overreach or unintended consequences. The Supreme Court ruling on probationary employees provides a temporary win for the administration's downsizing efforts on procedural grounds (standing), but the underlying legality remains contested in other courts. The administration appears determined to reshape the federal workforce rapidly, prioritizing perceived efficiency and ideological alignment over potential operational disruption.
"""
    text2 = """
Короче, жил-был маленький гномик. Он жил под ёлкой, и он любил мацутаки. Он каждое воскресенье ходил в лес, чтобы найти себе новую мацутаку, чтобы есть её целую неделю. Но он не мог иногда найти мацутаку, и тогда он целую неделю голодал. Когда ему становилось совсем тяжело, он находил что-то похожее на мацутаки, например, шампиньоны. Но они ему не нравились, потому что они были выращены на грядке, а не в лесу. А у мацутаков есть мицелий, и он общается со своими друзьями-мацутаками. И когда маленький гномик ел мацутаки, ему приходило снисхождение с небес. И он начинал очень-очень хорошо писать тексты и решать математические задачи. А когда он ел шампиньоны, он просто мог идти спать. А когда он вообще ничего не ел, он всё время спал и ждал свою следующую мацутаку.
"""
    start_time = time.time()
    # output_file = generate_audio(text1, "coqui_test.wav")
    output_file = generate_audio(text2, "coqui_anna_test.wav")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Audio generation took {elapsed_time:.2f} seconds")
    print(f"Final audio file: {output_file}")
    print("test completed.") 