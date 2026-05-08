# save as: rephraser.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from openai import OpenAI
import os
import random

# ---- Developer prompts (kept identical to your test script) ----
DEVELOPER_PROMPTS: Dict[str, str] = {
    "Drastic baseline": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. The content must remain unchanged, but the wording and sentence structure should be drastically revised.", #Drastic baseline
    "Moderate baseline": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. The content must remain unchanged, but the wording and sentence structure should be moderately revised.", #Moderate baseline
    "Informal": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Use informal, conversational language and replace chemical symbols with their common names whenever possible (e.g., Ag as silver).", #Informal tone
    "Order": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. The content must remain unchanged, but the order that details are shared in must be revised.", #Order
    "Error-heavy": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Make lots of spelling mistakes, grammar mistakes and inconsistent capitalization as yped quickly by a distracted user.", #Error-heavy
    "Error-light": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Add just a few spelling mistakes or typos, but keep it mostly correct.", #Error-light
    "Formal": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Use highly formal, technical, and academic language.", #Academic style
    "Concise": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Make the result as concise as possible (1-2 sentences).", #Concise summary
    "Detailed": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Expand it into a long and detailed version with extra descriptive phrases.", #Verbose
    "Instructional": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Write it as if you are giving direct requirements to a laboratory technician.", #Instructional 
    "First-person": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Write it in the first person, as if you are a researcher explaining a desired design's qualities.", #First-person
    "Passive": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Use passive voice throughout the rephrasing.", #Passive voice
    "Accessible": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Simplify the technical terms so that someone without a science background could understand.", #Accesible
    "Verbose": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Keep the meaning intact, but add redundancy, filler words, and a somewhat rambling style.", #Rambling
    "Persuasive": "Rephrase the following description of a multi-layer thin-film structure used for structural coloration. Write it in a persuasive marketing tone, subtly highlighting why the optical structure is exciting or innovative." #Persuasive
}

# Weighted choices (kept consistent with your script)
_OPTIONS: List[str] = [
    "Drastic baseline", "Moderate baseline", "Informal", "Order", "Error-heavy",
    "Error-light", "Formal", "Concise", "Instructional", "First-person",
    "Passive", "Accessible", "Persuasive", "Verbose", "Detailed"
]
_WEIGHTS: List[float] = [0.11, 0.11, 0.11, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.01, 0.01]


@dataclass(frozen=True)
class RephraseResult:
    text: str
    chosen_prompt_name: str
    model: str


class Rephraser:
    """
    Minimal helper for compile_datasets.py:
      - Deterministically picks a developer prompt using a provided random seed.
      - Sends the prompt + user text to the OpenAI API.
      - Returns ONLY the rephrased text (plus metadata if needed).
    Designed to be safe for parallel processes: no global RNG or shared client.
    """

    def __init__(
        self,
        model: str = "gpt-5-mini-2025-10-01",
        api_key_env: str = "CHROMA_KEY",
        timeout: Optional[float] = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set environment variable {api_key_env}"
            )
        # Fresh client per instance; fine for per-process parallelism.
        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    @staticmethod
    def _choose_prompt_name(seed: int) -> str:
        # Deterministic selection with user-provided seed
        rng = random.Random(seed)
        return rng.choices(_OPTIONS, weights=_WEIGHTS, k=1)[0]

    def rephrase(self, text: str, seed: int, prompt_name: Optional[str] = None) -> RephraseResult:
        """
        Rephrase `text` using a developer prompt chosen by `seed`.
        If `prompt_name` is provided, that prompt is used instead (seed ignored).
        Returns RephraseResult with the rephrased text and metadata.
        """
        chosen = prompt_name or self._choose_prompt_name(seed)
        dev_prompt = DEVELOPER_PROMPTS[chosen]

        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=1,
            messages=[
                {"role": "developer", "content": [{"type": "text", "text": dev_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": text}]},
            ],
        )
        out = resp.choices[0].message.content
        return RephraseResult(text=out, chosen_prompt_name=chosen, model=self._model)


# -------- Toy example (keep this so you can verify quickly) --------
if __name__ == "__main__":
    # Example input text you can replace while testing:
    example = (
        "Create an optical structure that consists of a four-layer stack; the stack may "
        "be composed of any material except GaInP, Al2O3, W, ITO, Ti, GaAs, Ni. "
        "At 30 degrees, the reflected color observed from this multilayer configuration "
        "must correspond to Lunar Green. If used, cSi must be thinner than 65 nm."
    )
    # Deterministic seed -> deterministic developer-prompt selection
    seed = 42

    r = Rephraser()
    result = r.rephrase(example, seed)
    print("== Chosen developer prompt ==", result.chosen_prompt_name)
    print("== Model ==", result.model)
    print("== Rephrased text ==\n", result.text)
