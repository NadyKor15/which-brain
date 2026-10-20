# Which Brain?

A hands-free kitchen helper for people with low vision. The phone is worn on the chest. Strands Decider 2B decides, for every phrase, whether it was meant for the helper at all, whether it is about medicine, and which Amazon Bedrock model should answer: a quick one, one that looks at the label, or one that reads closely.

Work in progress for the AWS Builder Center "Agent Experience Hackathon: Strands Decider".

## Rules the helper keeps (each one has a test)

- It says only text it can see. No "what such a can usually says".
- Three levels of confidence. Sure: it reads. Partly: "I can see the start: "Tomatoes in…", the rest is cut off, turn it." Unsure: "Hold it closer." It never says "I think" next to a number, a date, a weight or a temperature.
- Medicine: it may read the name printed on the label when it is sure ("The label says…"). It never states a dose. It offers to call your helper and dials only after you say yes.
- No action without "yes".
- A sound within 300 ms, "Looking…" if the answer takes longer than a second, never silent for more than 4 seconds.

## Credits

`src/decider_kit/` comes from RetailBox Automation's Decider toolkit (MIT, Copyright (c) 2026 Mikhail Oskola, see `src/decider_kit/LICENSE`). Notes on Strands interventions: `docs/STRANDS-INTERVENTIONS-NOTES.md`.

## License

MIT, see `LICENSE`.
