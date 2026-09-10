# ICAART 2027: verified submission requirements

Retrieved **9 September 2026**. Local preparation only; nothing has been submitted, registered, emailed or pushed by this task.

## Category, scope and dates

The [Call for Papers](https://icaart.scitevents.org/CallForPapers.aspx) lists Area 2, Artificial Intelligence, including Natural Language Processing, Knowledge Representation and Reasoning, Large Language Models (LLMs), and Visualization. The intended category is **Position Paper**, an empirical work-in-progress contribution.

The [Important Dates](https://icaart.scitevents.org/ImportantDates.aspx) page gives the Position Paper deadline as **22 October 2026, Anywhere on Earth (UTC−12)**. Notification: 4 December 2026; camera-ready and registration: 18 December 2026. Recheck before acting; local preparation is not an application or acceptance.

## Independent length and review constraints

The [Guidelines](https://icaart.scitevents.org/Guidelines.aspx) specify **8,000–40,000 characters excluding whitespace**, counting references, tables, graphs and appendices. Separately, accepted Position Papers have an ordinary **eight-page proceedings allowance**. This package targets at most eight formatted pages without paid extra pages. Anonymous PDF review is required. Remove identities and acknowledgments. Do not publicly post submitted manuscripts between initial submission and announcement of final selection results. The main paper discloses substantive AI assistance and cites the tool in the abstract, evaluation and provenance section. No identifying self-citations are included.

These submission-character and accepted-paper page constraints are verified separately in `verification.json`; neither substitutes for the other. The local character count includes searchable text inside all three vector figures and applies a conservative alternate bound. It is not an official PRIMORIS counter.

## Official template and measured layout

The [Templates page](https://icaart.scitevents.org/Templates.aspx) links the [official LaTeX archive](https://www.scitepress.org/documents/SCITEPRESS_Conference_Latex.zip). Downloaded archive SHA-256:

`ec6cfaa11962e08d5c6a402124f21c3bca3591397521406ab6d1889398a3807a`

The archive and complete example are retained under `vendor/`. Its four formatting files are copied byte-for-byte: `article.cls`, `SCITEPRESS.sty`, `apalike.sty`, `apalike.bst`. Their hashes are in `manifest.json`. None is edited.

The supplied example specifies A4, Times, a full-width title/abstract, two-column body, author-date references, 70–200 abstract words, 9-point captions, table captions above and figure captions below. Any appendix follows references without a forced new page. Body font is 10 points. Template margins are described as 26 mm left/right, 33 mm top and 42 mm bottom. The actual style uses 6.221 in = **158.0134 mm** text width and an 8 mm gap, giving **75.0067 mm** columns; the text-height and header expressions yield approximately 33.4 mm top and 42.604 mm bottom. The conference figures are natively reflowed at this exact width, minimum text 8.5 points, not reduced copies of the 178 mm manuscript plates.

## AI disclosure and supplementary-upload uncertainty

The AI-use page was rechecked on 9 September 2026. The concise provenance section identifies the tool, all affected manuscript parts, substantive roles and author responsibility without repeating a disclosure after every section.

The [AI Tools policy](https://icaart.scitevents.org/AiTools.aspx) requires transparent naming and description of substantive AI assistance; AI tools cannot be authors and humans remain responsible. Codex assisted code, annotation, assessment, figures and manuscript writing; Qwen generated the experimental records. The main paper explicitly discloses these roles. No assertion is made that Codex assistance was limited to editing. The cited Codex announcement identifies the tool, not a known backend version for every historical interaction.

There is a placement ambiguity: the general guidelines direct AI disclosure to acknowledgments while directing their omission for anonymous review; the AI policy also mentions an appropriate disclosure section. We use a non-identifying AI-use section and retain an author-version acknowledgment separately, pending confirmation.

No explicit Position Paper supplementary-upload permission was found in the required pages, the [conference FAQ](https://icaart.scitevents.org/FAQ.aspx), or the public [PRIMORIS entry page](https://www.insticc.org/Primoris/Default.aspx/). The guidelines' complementing-material permission concerns the **Abstracts track**, not this category. No logged-in upload workflow was accessed. The companion is therefore **locally prepared supporting material; submission eligibility unconfirmed**. The main paper stands alone.

## Prior public version and presentation

The [FAQ](https://icaart.scitevents.org/FAQ.aspx), section 3.8, distinguishes archival publication from some non-archival prior work and permits certain non-archival republication. This does not resolve anonymity for this already-public manuscript. Its presentation section requires presentation for publication and recommends contacting the secretariat if neither author nor substitute can attend; it does not establish remote-presentation permission. `author_actions.md` records the chronology, and `secretariat_inquiry_unsent.md` asks for case-specific clarification. Neither automatic eligibility nor automatic disqualification is assumed.
