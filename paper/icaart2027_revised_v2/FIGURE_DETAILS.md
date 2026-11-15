# Original figure details retained with this revision

The main captions were shortened in response to the reviews. The following original LaTeX caption text preserves the executed-question macros, display omissions and interpretation notes. Exact questions are defined in `generated/numbers.tex` and retained in the public request payloads. Figure 1 now displays ten source sentences instead of three; its graph content is unchanged.

## fig:fable

```tex
Fable actions from pre-extract/select (A) and independent contextual extraction (B). + indicates supported meaning and $\sim$ partial meaning. Display: five of eight A records and all five B records. A's awakening, hunter capture and rope binding are omitted only from display. All predictions remain scored. Executed question: \FableQuestion\ Source display excerpts S2 and S5 from the complete supplied fable. The spare edge loses conditional scope and retains the misspelled key \texttt{valid until}. B loses the face-specific running endpoint and omits the early capture/release. Other displayed qualifications are null. Assessments are Codex-authored.
```

## fig:orchard

```tex
Orchard: the actual model extraction before the question (top network), followed by fixed selection from that extraction (bottom). Executed question: \OrchardQuestion\ All 14 extracted records and all five selected records are shown. Source excerpts S1, S3 and S11 come from the complete fourteen-sentence input. Exact endpoint strings share nodes. The selected six-node network is connected. Holder/attitude badges qualify two location edges without adding relationships. The broad extraction is not a gold graph. + denotes retained support for the selected records, not a selection colour. Node categories are display annotations. Horizontal anchors are shared, with compressed vertical spacing below.
```

## fig:alice

```tex
Alice: independent contextual extraction (B) for two requests over identical complete evidence. Display: two of five action edges (two sitting edges and Rabbit running omitted) and two of three claims (book content omitted). All remain scored. Executed action question: \AliceActionQuestion\ Executed claims question: \AliceClaimQuestion\ Source excerpts are marked. + indicates supported meaning, x an unsupported participant, and ? unresolved consideration. The thought remains the actual sentence-valued object. All bounds are null. The final record has holder Alice and null attitude.
```
