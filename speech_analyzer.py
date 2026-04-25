import os
import torch
import gradio as gr

from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


# -----------------------------------------
# Watsonx Credentials and Model Parameters
# -----------------------------------------

my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,   # Max tokens generated per response
    GenParams.TEMPERATURE: 0.1       # Lower = more deterministic output
}


# -----------------------------------------
# Watsonx LLaMA Model Setup
# -----------------------------------------

llama_model = Model(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(llama_model)


# -----------------------------------------
# Prompt Template & LangChain Setup
# -----------------------------------------

template = """
<>
List the key points with details from the context:
[INST]
The context:
{context}
[/INST]
<>
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=template
)

llama_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)


# -----------------------------------------
# Speech-to-Text + LLM Processing
# -----------------------------------------

def transcript_audio(audio_file):
    """
    Transcribes audio using Whisper
    and summarizes key points using LLaMA via Watsonx
    """

    # Speech recognition pipeline
    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    # Transcribe audio
    transcript_text = asr_pipe(
        audio_file,
        batch_size=8
    )["text"]

    # Process transcript using Watsonx LLM
    result = llama_chain.run(transcript_text)

    return result


# -----------------------------------------
# Gradio Interface
# -----------------------------------------

audio_input = gr.Audio(
    sources="upload",
    type="filepath"
)

output_text = gr.Textbox()

interface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload an audio file for transcription and summarization"
)

interface.launch(
    server_name="0.0.0.0",
    server_port=7860
)
