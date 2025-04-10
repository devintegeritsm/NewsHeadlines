from kokoro import KPipeline
# from IPython.display import display, Audio
import soundfile as sf
import torch
import time
import numpy as np
import os
import uuid
import warnings

# Filter out specific PyTorch warnings
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`")

def concatenate_wav_files(base_filename, num_files, output_filename):
    """Concatenate multiple WAV files into a single file."""
    audio_data = []
    sample_rate = None
    
    for i in range(num_files):
        temp_filename = f'{base_filename}_{i}.wav'
        data, rate = sf.read(temp_filename)
        audio_data.append(data)
        if sample_rate is None:
            sample_rate = rate
        # Clean up temporary file
        os.remove(temp_filename)
    
    # Concatenate all audio segments
    combined_audio = np.concatenate(audio_data)
    # Write to output file
    sf.write(output_filename, combined_audio, sample_rate)
    return output_filename


def generate_audio(text: str, output_filename: str):
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    generator = pipeline(text, voice='af_heart')
    segments = 0
    
    temp_filename_base = str(uuid.uuid4())
    for i, (gs, ps, audio) in enumerate(generator):
        # print(i, gs, ps)
        # display(Audio(data=audio, rate=24000, autoplay=i==0))
        temp_filename = f'{temp_filename_base}_{i}'
        sf.write(f'{temp_filename}.wav', audio, 24000)
        segments = i + 1
    
    # Concatenate all segments into the final file
    if segments > 0:
        concatenate_wav_files(temp_filename_base, segments, output_filename)
    
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
    output_file = generate_audio(text1, "kokoro_test.wav")
    # output_file = generate_audio(text2, "kokoro_anna_test.wav")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Audio generation took {elapsed_time:.2f} seconds")
    print(f"Final audio file: {output_file}")
    print("test completed.") 