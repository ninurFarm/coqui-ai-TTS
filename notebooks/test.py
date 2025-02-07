from pprint import pprint as print
from typing import Literal

import torch

from TTS.api import TTS

device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
print(object=f"Using device: {device}")


tts: TTS = TTS(
    model_path="./models/v1/as/fastpitch/best_model.pth",
    config_path="./models/v1/as/fastpitch/config.json",
    vocoder_path="./models/v1/as/hifigan/best_model.pth",
    vocoder_config_path="./models/v1/as/hifigan/config.json",
).to(device=device)

wav: str = tts.tts_to_file(
    text="""হেৰুৱা স্মৃতিৰ গান

কুঁৱলীৰ দৰে মন, হেৰাই যোৱা সপোন,
নিৰৱতাৰ গান,  বুকুৰ শূন্যস্থান।
শেৱালিৰ সুবাস,  উৰি গ'ল দূৰলৈ,
একাডেমীয়া ৰাতি,  চন্দ্ৰৰ নীৰৱতাই।

নদীৰ সোঁতত,  ভাসি গ'ল সময়,
হেৰুৱা প্ৰেমৰ,  অশ্ৰুৰ  নদী বয়।
স্মৃতিৰ  গভীৰত,  জীৱনৰ  গান,
হেৰুৱা  সময়ৰ,  বিলাপৰ  তান।

শূন্য  হাত,  শূন্য  মন,  শূন্য  আশা,
হেৰুৱা  প্ৰেমৰ,  এয়াই  ভাষা।
তথাপিও  জীৱন,  চলিয়েই  থাকে,
নতুন  সপোনে,  আকৌ  মাতে।""",
    speaker="male",
    file_path="output.wav",
)
