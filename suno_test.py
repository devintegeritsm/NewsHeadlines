import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ["SUNO_ENABLE_MPS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"
os.environ["SUNO_USE_SMALL_MODELS"] = "False"

import warnings
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
import time
import os
import torch
import soundfile as sf
import nltk

# nightly cpu for mps
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu


TMP_FOLDER = 'tmp'
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

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


def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        print("NLTK 'punkt' tokenizer not found. Downloading...")
        nltk.download('punkt')
        print("'punkt' downloaded.")
    try:
        nltk.data.find('punkt_tab')
    except:
        print("NLTK 'punkt_tab' tokenizer not found. Downloading...")
        nltk.download('punkt_tab')
        print("'punkt_tab' downloaded.")

def split_text_into_sentences(text):
    """Splits text into sentences using NLTK."""
    try:
         # Ensure punkt is available before trying to use it
        nltk.data.find('tokenizers/punkt')
    except:
        print("Error: NLTK 'punkt' model not found. Please run nltk.download('punkt')")
        # Optionally, try to download it here if you want the script to be self-contained
        # print("Attempting to download 'punkt' now...")
        # nltk.download('punkt')
        # print("'punkt' downloaded.")
        # return [] # Or raise an error if download fails
        raise RuntimeError("NLTK 'punkt' model required but not found.") # More explicit failure

    print("Using NLTK for sentence tokenization.")
    sentences = nltk.sent_tokenize(text.strip())
    print(f"Split text into {len(sentences)} sentences.")
     # Basic check for very long "sentences" - might indicate splitting failed
    for i, s in enumerate(sentences):
        if len(s) > 500: # Arbitrary threshold, adjust as needed
            print(f"Warning: Chunk {i+1} seems long ({len(s)} chars). Consider further splitting: '{s[:100]}...'")
    return sentences

def check_mps_availability():
    """Checks if MPS is available and prints status."""
    # ... (same as before) ...
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("❌ MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
            return False
        else:
            print("❌ MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
            return False
    print("✅ MPS is available on this device.")
    return True

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

def generate_long_tts_with_mps_tempfiles(text_prompt: str, speaker: str, output_filename: str, force_cpu: bool = False):
    """
    Generates TTS for long text using Bark by splitting, saving chunks to
    temporary files, and concatenating with ffmpeg. Attempts MPS acceleration.

    Args:
        text_prompt (str): The text to synthesize (can be long).
        speaker (str): The voice prompt/speaker ID to use.
        output_filename (str): The name of the final output WAV file.
        force_cpu (bool): If True, forces usage of CPU.
    """
    print("=" * 40)
    print(" Bark Long Text TTS (Temp Files & ffmpeg)")
    print("=" * 40)

    # --- Device Selection ---
    use_mps_if_available = not force_cpu and check_mps_availability()
    selected_device = "mps" if use_mps_if_available else "cpu"
    # ... (rest of device selection logic - setting SUNO_ENABLE_MPS env var) ...
    if use_mps_if_available:
        print("🚀 Attempting to use MPS acceleration.")
        os.environ['SUNO_ENABLE_MPS'] = 'True'
    else:
        if force_cpu:
            print("🐌 CPU usage forced.")
        else:
            print("🐌 MPS not available or check failed. Using CPU.")
        os.environ['SUNO_ENABLE_MPS'] = 'False'
    print(f"Selected compute device: {selected_device.upper()}")

    # --- Model Loading ---
    print("\nLoading Bark models...")
    # ... (same model loading logic as before) ...
    print("(This may take a while and download ~5GB+ on the first run)")
    start_load_time = time.time()
    use_small = False
    try:
        preload_models(
            text_use_gpu=use_mps_if_available, text_use_small=use_small,
            coarse_use_gpu=use_mps_if_available, coarse_use_small=use_small,
            fine_use_gpu=use_mps_if_available, fine_use_small=use_small,
            codec_use_gpu=use_mps_if_available, force_reload=True
        )
    except TypeError:
         print("\nNote: Your Bark version's preload_models might not accept GPU flags.")
         print("Relying on SUNO_ENABLE_MPS environment variable for device selection.")
         preload_models()
    load_time = time.time() - start_load_time
    print(f"Models loaded in {load_time:.2f} seconds.")

    # --- Text Splitting ---
    print("\nSplitting text into manageable chunks...")
    text_chunks = split_text_into_sentences(text_prompt)
    if not text_chunks:
        print("Error: Text splitting resulted in no chunks.")
        return

    # --- Processing with Temporary Files ---
    temp_file_paths = []
    total_gen_time = 0
    temp_filename_base = os.path.join(TMP_FOLDER,"temp_chunk")
    print(f"\nGenerating audio chunk by chunk ({len(text_chunks)} chunks) and saving to temporary files...")

    temp_dir = TMP_FOLDER

    for i, chunk in enumerate(text_chunks):
        print(f"--- Generating chunk {i+1}/{len(text_chunks)} ---")
        print(f"Text: '{chunk}'")
        start_gen_time = time.time()
        temp_wav_path = f'{temp_filename_base}_{i}.wav'
        print(f"File: '{temp_wav_path}'")


        try:
            audio_array = generate_audio(
                chunk,
                history_prompt=speaker,
                text_temp=0.7,
                waveform_temp=0.7,
                silent=True # Optional: Uncomment if available/desired
            )

            # Convert to int16 and save WAV chunk
            audio_int16 = (audio_array * 32767).astype(np.int16)
            write_wav(temp_wav_path, SAMPLE_RATE, audio_int16)
            temp_file_paths.append(temp_wav_path) # Store path for ffmpeg list

            gen_time = time.time() - start_gen_time
            total_gen_time += gen_time
            print(f"Chunk {i+1} generated and saved to '{os.path.basename(temp_wav_path)}' in {gen_time:.2f} seconds.")

            # Clear large objects from memory (optional, Python's GC usually handles it)
            del audio_array
            del audio_int16

        except Exception as e:
            print(f"🚨 Error generating audio for chunk {i+1}: {e}")
            print("Skipping this chunk.")
            if os.path.exists(temp_wav_path): # Clean up partially created file if error occurred after path generation
                try:
                    os.remove(temp_wav_path)
                except OSError:
                        print(f"Warning: Could not remove temporary file {temp_wav_path} after error.")
            # Decide if you want to stop or continue on error
            # continue # or break


    if len(text_chunks) > 0:
        concatenate_wav_files(temp_filename_base, len(text_chunks), output_filename)
        

# --- Main Execution ---
if __name__ == "__main__":

    download_nltk()
    # --- Configuration ---
    sample_text = (
        "Hello there! This is a demonstration of the Bark text-to-speech model "
        "running with Metal Performance Shaders acceleration on an Apple Silicon Mac. "
        "Hopefully, this sounds reasonably good and generates quickly!"
    )

    sample_text = """
Короче, жил-был маленький гномик. Он жил под ёлкой, и он любил мацутаки. Он каждое воскресенье ходил в лес, чтобы найти себе новую мацутаку, чтобы есть её целую неделю. Но он не мог иногда найти мацутаку, и тогда он целую неделю голодал. Когда ему становилось совсем тяжело, он находил что-то похожее на мацутаки, например, шампиньоны. Но они ему не нравились, потому что они были выращены на грядке, а не в лесу. А у мацутаков есть мицелий, и он общается со своими друзьями-мацутаками. И когда маленький гномик ел мацутаки, ему приходило снисхождение с небес. И он начинал очень-очень хорошо писать тексты и решать математические задачи. А когда он ел шампиньоны, он просто мог идти спать. А когда он вообще ничего не ел, он всё время спал и ждал свою следующую мацутаку.
"""

    output_file = "suno_anna_test.wav"
    # Try different speakers from the Bark documentation if you like
    # e.g., 'v2/en_speaker_1', 'v2/fr_speaker_5', 'v2/zh_speaker_3' etc.
    # selected_speaker = "v2/en_speaker_9"
    selected_speaker = "v2/ru_speaker_4"

    # Set force_cpu=True if you want to compare performance or if MPS causes issues
    force_cpu_mode = False

    # Run the generation function
    generate_long_tts_with_mps_tempfiles(
        text_prompt=sample_text,
        speaker=selected_speaker,
        output_filename=output_file,
        force_cpu=force_cpu_mode
    )

    print("\nScript finished.")