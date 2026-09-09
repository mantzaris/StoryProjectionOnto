/* Evaluate in the generated index through the existing local Chrome/CDP harness. */
(async () => {
  await document.fonts.ready;
  const records = JSON.parse(document.getElementById("edge-data").textContent);
  let checked = 0;
  for (const [id, record] of Object.entries(records)) {
    const el = document.querySelector(`[data-record="${id}"]`);
    if (!el) throw Error("Missing rendered record " + id);
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    const actual = [...document.querySelectorAll(".cited")].map(e => e.id).sort();
    const expected = record.fact.evidence_ids.map(
      e => `evidence-${record.dataset}-${record.story}-${e}`
    ).sort();
    if (JSON.stringify(actual) !== JSON.stringify(expected)) throw Error("Wrong evidence " + id);
    const box = document.getElementById("inspector");
    if (!box.textContent.includes("Assessed status: " + record.status)) throw Error("Missing status");
    if (!box.textContent.includes("citation link only")) throw Error("No citation caveat");
    if (JSON.stringify(JSON.parse(box.querySelector("pre").textContent)) !== JSON.stringify(record.fact)) {
      throw Error("Inspector changed actual record " + id);
    }
    checked++;
  }
  const first = document.querySelector("[data-record]");
  first.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
  if (!first.classList.contains("selected")) throw Error("Keyboard selection failed");
  const ids = [...document.querySelectorAll("[id]")].map(e => e.id);
  if (new Set(ids).size !== ids.length) throw Error("Duplicate inline SVG identifiers");
  const failures = [...document.fonts].filter(f => f.status !== "loaded");
  if (failures.length) throw Error("Embedded fonts did not load");
  return {checked_edges: checked, keyboard: "pass", cited_evidence: "exact", fonts: document.fonts.size};
})()
