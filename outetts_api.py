import outetts
import os
import sys
import time  # Added for timing functionality

# --- Set the Environment Variable ---
# Set it to 'true' to enable parallelism
# Set it to 'false' to disable parallelism (useful to avoid warnings/errors in some environments)
variable_name = 'TOKENIZERS_PARALLELISM'
desired_value = 'true' # Or 'false'

print(f"[{sys.argv[0]}] Before setting: {variable_name}={os.environ.get(variable_name)}")
os.environ[variable_name] = desired_value
print(f"[{sys.argv[0]}] After setting:  {variable_name}={os.environ.get(variable_name)}")

# Initialize the interface
interface = outetts.Interface(
    config=outetts.ModelConfig.auto_config(
        model=outetts.Models.VERSION_1_0_SIZE_1B,
        # For llama.cpp backend
        backend=outetts.Backend.LLAMACPP,
        quantization=outetts.LlamaCppQuantization.FP16,
        # For transformers backend
        # backend=outetts.Backend.HF,
    )
)

# Load the default speaker profile
speaker = interface.load_default_speaker("EN-FEMALE-1-NEUTRAL")

    # Or create your own speaker profiles in seconds and reuse them instantly
    # speaker = interface.create_speaker("path/to/audio.wav")
    # interface.save_speaker(speaker, "speaker.json")
    # speaker = interface.load_speaker("speaker.json")

def generate_audio(text: str, filename: str):
    try:
        # Generate speech
        output = interface.generate(
            config=outetts.GenerationConfig(
                text=text,
            generation_type=outetts.GenerationType.CHUNKED,
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(
                temperature=0.4
            ),
            )
        )

    # Save to file
        output.save(filename)

    except Exception as e:
        if "TypeError: 'NoneType' object is not callable" in str(e):
            print(".")
        else:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Starting outetts test...")
    # text = "Hello, how are you doing?"
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
    # generate_audio(text, "outetts_test.wav")
    generate_audio(text2, "anna_test.wav")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Audio generation took {elapsed_time:.2f} seconds")
    print("outetts test completed.") 