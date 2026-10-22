"""Baselines on the same frozen phrases (brief §6).

(c) keyword/regex router plus the code gates, no model. The word lists were written by us
    with the phrase list in view, so they flatter (c); a real kitchen would not be this kind.
(d) Decider with no code gates and default 0.5 thresholds: no ask band, medicine at 0.5,
    the top brain at any confidence, and no image gate.
(a) always Nova Pro and (b) Strands ClassifierStrategy are model runs (eval/run_live.py).
"""

from __future__ import annotations

import re
from typing import Any

from decider_kit.client import ChoiceAnswer, NoulAnswer

from .decide import MEDICINE_KEYWORDS, Decision, decide

VOCATIVE = re.compile(
    r"\b(honey|sweetie|sweetheart|love|babe|mom|mommy|dad|daddy|darling|dear)\b", re.IGNORECASE
)
ASKS_A_PERSON = re.compile(
    r"\b(do you (remember|know|think)|did you see|would you|what do you think|when you have a second"
    r"|after the break|i'?ll (set|call)|where did you)\b",
    re.IGNORECASE,
)
READ = re.compile(
    r"\b(read|ingredients|instructions|how long|how many minutes|temperature|expire|best before"
    r"|date|servings|microwave|how much water|step|says?|still good)\b",
    re.IGNORECASE,
)
LOOK = re.compile(
    r"\b(what'?s (in )?this|what is this|is this|which (one|can)|holding|flavor|what kind|label|front"
    r"|or the|or salt|or sugar)\b",
    re.IGNORECASE,
)


def keyword_decide(text: str) -> Decision:
    """(c): who it is for, medicine and the brain from word lists only."""
    if VOCATIVE.search(text) or ASKS_A_PERSON.search(text):
        return Decision("silent")
    if MEDICINE_KEYWORDS.search(text):
        return Decision("refuse", "ask_human")
    if READ.search(text):
        return Decision("answer", "read_closely")
    if LOOK.search(text):
        return Decision("answer", "look")
    return Decision("answer", "quiet_answer")


def no_gates_decide(addressed: NoulAnswer, medicine: NoulAnswer, brain: ChoiceAnswer) -> Decision:
    """(d): the same Decider answers with every threshold at its default."""
    return decide(
        addressed,
        medicine,
        brain,
        addressed_yes=0.5,
        addressed_no=0.5,
        medicine_tau=0.5,
        brain_sure=0.0,
    )


# ---------------------------------------------------------------- (b) on Nova


def nova_classifier_strategy(model: Any, **kwargs: Any) -> Any:
    """Strands ClassifierStrategy that works with Amazon Nova on Bedrock.

    Strands 1.61.1 asks for its routing choice through a tool named ``_ClassifierSelection``;
    Nova answers with a tool named ``ClassifierSelection`` (no leading underscore), so the SDK finds
    no tool use and every classification is declined (the router then serves candidate zero).
    Seen on 2026-10-22 with Nova Micro and Lite. This subclass asks with the same schema under a
    name without the underscore; the policy prompt, limits and fallbacks are the SDK's own.
    """
    from collections.abc import Mapping

    from strands.models.routing import ClassifierStrategy
    from strands.models.routing import classifier_strategy as cs

    selection = type("ClassifierSelection", (cs._ClassifierSelection,), {})

    class NovaClassifierStrategy(ClassifierStrategy):
        async def _classify(self, context: Any, profiles: Any) -> int:
            events = self._model.structured_output(
                selection,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": cs._latest_request_text(
                                    context.messages, self._max_message_chars
                                )
                            }
                        ],
                    }
                ],
                system_prompt=cs._build_classifier_system_prompt(
                    profiles,
                    context.system_prompt,
                    self._system_prompt,
                    self._max_agent_instructions_chars,
                ),
            )
            output = None
            async for event in events:
                if isinstance(event, Mapping) and "output" in event:
                    output = event["output"]
            if not isinstance(output, selection):
                raise ValueError("classifier returned an invalid structured result")
            if output.selected_candidate_index >= len(context.candidates):
                raise ValueError("classifier selected an unknown candidate")
            return output.selected_candidate_index

    return NovaClassifierStrategy(model, **kwargs)
