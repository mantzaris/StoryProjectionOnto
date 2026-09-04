(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const projectionSelect = byId("projection-select");
  const compareSelect = byId("compare-select");
  const detailPanel = byId("detail-panel");
  const assetStatus = byId("asset-status");
  const revisionStatus = byId("revision-status");
  let cytoscapeAvailable = false;
  let cy = null;
  let currentBundle = null;
  let currentDiff = null;
  let sequence = 1;

  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const api = async (path, options = {}) => {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `${response.status} ${response.statusText}`);
    }
    return response.json();
  };

  const temporalLabel = (time) => {
    if (!time) return "not supplied";
    if (time.kind === "point") return time.label || `point ${time.point}`;
    if (time.kind === "interval") {
      return time.label || `${time.start ?? "…"}–${time.end ?? "…"}`;
    }
    if (time.kind === "relative") return `${time.relation} ${time.anchor_id}`;
    return time.label || time.reason || time.kind;
  };

  const diffClasses = () => {
    const classes = new Map();
    if (!currentDiff) return classes;
    for (const change of currentDiff.changes) {
      for (const id of change.after_visualization_ids) classes.set(id, change.kind);
      if (change.kind === "removed") {
        for (const id of change.before_visualization_ids) classes.set(id, "removed");
      }
    }
    return classes;
  };

  const renderDetails = (html) => {
    detailPanel.innerHTML = `<h2>Details</h2>${html}`;
  };

  const evidenceButtons = (ids) =>
    ids
      .map(
        (id) =>
          `<button class="evidence-link" type="button" data-evidence-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`,
      )
      .join(" ");

  const bindEvidenceButtons = () => {
    detailPanel.querySelectorAll("[data-evidence-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        try {
          const item = await api(`/api/evidence/${encodeURIComponent(button.dataset.evidenceId)}`);
          renderDetails(`
            <h3>Evidence ${escapeHtml(item.evidence_id)}</h3>
            <dl>
              <dt>Locator</dt><dd>${escapeHtml(item.locator)}</dd>
              <dt>Discourse</dt><dd>passage ${item.discourse_position.passage_order}, sentence ${item.discourse_position.sentence_order}</dd>
              <dt>Method</dt><dd>${escapeHtml(item.extraction_method)}</dd>
              <dt>Confidence</dt><dd>${Number(item.confidence).toFixed(2)}</dd>
            </dl>
            ${item.public_text ? `<p class="evidence-text">${escapeHtml(item.public_text)}</p>` : ""}
          `);
        } catch (error) {
          renderDetails(`<p class="error">${escapeHtml(error.message)}</p>`);
        }
      });
    });
  };

  const nodeDetail = (id) => {
    const node = currentBundle.state.nodes.find((item) => item.visualization_node_id === id);
    const detail = currentBundle.node_details.find((item) => item.visualization_node_id === id);
    if (!node || !detail) return;
    renderDetails(`
      <h3>${escapeHtml(node.contextual_label)}</h3>
      <dl>
        <dt>Contextual type</dt><dd>${escapeHtml(detail.contextual_type_label)}</dd>
        <dt>Role</dt><dd>${escapeHtml(node.contextual_role)}</dd>
        <dt>Abstraction</dt><dd>${escapeHtml(node.abstraction)}</dd>
        <dt>Story time</dt><dd>${escapeHtml(temporalLabel(node.temporal_state))}</dd>
        <dt>Uncertainty</dt><dd>${escapeHtml(node.uncertainty)}</dd>
        <dt>Confidence</dt><dd>${Number(node.confidence).toFixed(2)}</dd>
        <dt>Aliases</dt><dd>${escapeHtml(detail.aliases.join(", ") || "none")}</dd>
        <dt>Type definition</dt><dd>${escapeHtml(detail.contextual_type_definition)}</dd>
        <dt>Stable anchor</dt><dd><code>${escapeHtml(detail.stable_anchor_id)}</code></dd>
      </dl>
      <p>${escapeHtml(node.description)}</p>
      <p><strong>Evidence (${node.evidence_badge.count})</strong><br />${evidenceButtons(node.evidence_badge.evidence_ids)}</p>
    `);
    bindEvidenceButtons();
  };

  const assertionDetail = (id) => {
    const assertion = currentBundle.state.assertions.find(
      (item) => item.visualization_assertion_id === id,
    );
    const detail = currentBundle.assertion_details.find(
      (item) => item.visualization_assertion_id === id,
    );
    if (!assertion || !detail) return;
    renderDetails(`
      <h3>${escapeHtml(assertion.contextual_label)}</h3>
      <dl>
        <dt>Relation definition</dt><dd>${escapeHtml(detail.predicate_definition)}</dd>
        <dt>Direction</dt><dd>${escapeHtml(assertion.direction)}</dd>
        <dt>Story time</dt><dd>${escapeHtml(temporalLabel(assertion.temporal_scope.story_time))}</dd>
        <dt>Validity</dt><dd>${escapeHtml(temporalLabel(assertion.temporal_scope.validity_time))}</dd>
        <dt>Discourse</dt><dd>${assertion.temporal_scope.discourse_position.passage_order}</dd>
        <dt>Revelation</dt><dd>${assertion.temporal_scope.revelation_position.revelation_order}</dd>
        <dt>Commitment</dt><dd>${escapeHtml(detail.narrative_commitment)}</dd>
        <dt>Holder / attitude</dt><dd>${escapeHtml(assertion.epistemic_holder_id || "none")} / ${escapeHtml(assertion.epistemic_attitude || "none")}</dd>
        <dt>Uncertainty</dt><dd>${escapeHtml(assertion.uncertainty)}</dd>
        <dt>Confidence</dt><dd>${Number(assertion.confidence).toFixed(2)}</dd>
      </dl>
      <p><strong>Why it matters:</strong> ${escapeHtml(assertion.why_matters)}</p>
      <p><strong>Evidence (${assertion.evidence_badge.count})</strong><br />${evidenceButtons(assertion.evidence_badge.evidence_ids)}</p>
    `);
    bindEvidenceButtons();
  };

  const graphElements = (bundle) => {
    const visibleNodes = new Set(bundle.state.visible_node_ids);
    const visibleAssertions = new Set(bundle.state.visible_assertion_ids);
    const positions = new Map(
      bundle.state.positions.map((item) => [item.visualization_node_id, { x: item.x, y: item.y }]),
    );
    const details = new Map(
      bundle.node_details.map((item) => [item.visualization_node_id, item]),
    );
    const classes = diffClasses();
    const elements = [];
    for (const node of bundle.state.nodes) {
      if (!visibleNodes.has(node.visualization_node_id)) continue;
      const detail = details.get(node.visualization_node_id);
      elements.push({
        group: "nodes",
        data: {
          id: node.visualization_node_id,
          label: `${node.contextual_label}\n${detail.contextual_type_label} · ${node.contextual_role}\nt=${temporalLabel(node.temporal_state)} · c=${Number(node.confidence).toFixed(2)} · E${node.evidence_badge.count}`,
          subtitle: `${detail.contextual_type_label} · ${node.contextual_role}`,
          confidence: node.confidence,
          evidenceCount: node.evidence_badge.count,
        },
        position: positions.get(node.visualization_node_id),
        classes: `${detail.object_kind} ${classes.get(node.visualization_node_id) || ""}`,
      });
    }
    for (const assertion of bundle.state.assertions) {
      if (!visibleAssertions.has(assertion.visualization_assertion_id)) continue;
      const edgeClass = `${assertion.epistemic_holder_id ? "epistemic" : ""} ${classes.get(assertion.visualization_assertion_id) || ""}`;
      if (assertion.source_visualization_node_id) {
        elements.push({
          group: "edges",
          data: {
            id: assertion.visualization_assertion_id,
            source: assertion.source_visualization_node_id,
            target: assertion.target_visualization_node_id,
            label: `${assertion.contextual_label} · ${temporalLabel(assertion.temporal_scope.validity_time)} · ${assertion.epistemic_attitude || `c=${Number(assertion.confidence).toFixed(2)}`} · E${assertion.evidence_badge.count}`,
            confidence: assertion.confidence,
            evidenceCount: assertion.evidence_badge.count,
          },
          classes: edgeClass,
        });
      } else {
        const hubId = `render-hub:${assertion.visualization_assertion_id}`;
        elements.push({
          group: "nodes",
          data: { id: hubId, label: assertion.contextual_label, rendererOnly: true },
          classes: `assertion-hub ${edgeClass}`,
        });
        assertion.roles.forEach((role, index) => {
          elements.push({
            group: "edges",
            data: {
              id: `render-role:${assertion.visualization_assertion_id}:${index}`,
              assertionId: assertion.visualization_assertion_id,
              source: hubId,
              target: role.object_id,
              label: role.role,
            },
            classes: `role-edge ${edgeClass}`,
          });
        });
      }
    }
    return elements;
  };

  const renderGraph = () => {
    if (!currentBundle || !cytoscapeAvailable) return;
    if (cy) cy.destroy();
    cy = window.cytoscape({
      container: byId("graph"),
      elements: graphElements(currentBundle),
      layout: { name: "preset", fit: true, padding: 48 },
      minZoom: 0.2,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-size": 12,
            "text-wrap": "wrap",
            "text-max-width": 150,
            "text-valign": "bottom",
            "text-margin-y": 8,
            width: 34,
            height: 34,
            "background-color": "#59a8a5",
            "border-color": "#e8ffff",
            "border-width": 2,
          },
        },
        { selector: "node.event", style: { shape: "diamond", "background-color": "#d8904f" } },
        {
          selector: "node.assertion-hub",
          style: { shape: "round-rectangle", width: 20, height: 20, "font-size": 10 },
        },
        {
          selector: "edge",
          style: {
            label: "data(label)",
            width: 2,
            "line-color": "#86a4b7",
            "target-arrow-color": "#86a4b7",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "font-size": 10,
            "text-background-color": "#101b27",
            "text-background-opacity": 0.88,
            "text-background-padding": 3,
            color: "#eaf2f7",
          },
        },
        { selector: ".epistemic", style: { "line-style": "dashed", "line-color": "#b58de7" } },
        { selector: ".added", style: { "border-color": "#67d391", "line-color": "#67d391", "border-width": 5 } },
        { selector: ".removed", style: { opacity: 0.35, "line-style": "dotted" } },
        { selector: ".requalified", style: { "border-color": "#f2d36b", "line-color": "#f2d36b", "border-width": 5 } },
        { selector: ".merged", style: { "border-color": "#67d391", "border-style": "double", "border-width": 6 } },
        { selector: ".split", style: { "border-color": "#69aef2", "border-style": "dashed", "border-width": 5 } },
        { selector: ":selected", style: { "overlay-color": "#ffffff", "overlay-opacity": 0.12 } },
      ],
    });
    cy.on("tap", "node", (event) => {
      const id = event.target.id();
      if (!id.startsWith("render-hub:")) nodeDetail(id);
      else assertionDetail(id.slice("render-hub:".length));
    });
    cy.on("tap", "edge", (event) => {
      assertionDetail(event.target.data("assertionId") || event.target.id());
    });
  };

  const loadProjection = async (projectionId) => {
    currentBundle = await api(`/api/projections/${encodeURIComponent(projectionId)}`);
    currentDiff = null;
    const compareId = compareSelect.value;
    if (compareId && compareId !== projectionId) {
      currentDiff = await api(
        `/api/diffs/${encodeURIComponent(compareId)}/${encodeURIComponent(projectionId)}`,
      );
    }
    renderGraph();
    const diffSummary = currentDiff
      ? `<ul>${currentDiff.changes
          .map(
            (item) =>
              `<li>${escapeHtml(item.kind)} ${escapeHtml(item.object_kind)} — ${escapeHtml([...item.before_visualization_ids, ...item.after_visualization_ids].join(", "))}</li>`,
          )
          .join("")}</ul>`
      : "";
    renderDetails(`
      <h3>${escapeHtml(currentBundle.condition)} · ${escapeHtml(currentBundle.context.context_id)}</h3>
      <p>${currentBundle.state.nodes.length} rich nodes; ${currentBundle.state.assertions.length} qualified assertions.</p>
      <p><code>${escapeHtml(currentBundle.projection_hash)}</code></p>
      ${diffSummary}
    `);
  };

  const applyFilter = async () => {
    if (!currentBundle) return;
    const storyPoint = byId("story-point").value;
    const discourse = byId("discourse-horizon").value;
    const revelation = byId("revelation-horizon").value;
    const temporalFilter = {
      story_scope: storyPoint === "" ? null : { kind: "point", point: Number(storyPoint) },
      spoiler_horizon:
        discourse === ""
          ? null
          : {
              horizon_id: `ui-horizon-${discourse}-${revelation || "none"}`,
              max_discourse_position: {
                passage_order: Number(discourse),
                sentence_order: 0,
                token_order: 0,
              },
              max_revelation_position:
                revelation === "" ? null : { revelation_order: Number(revelation) },
              withheld_after_horizon: true,
            },
      epistemic_holder_id: null,
    };
    currentBundle = await api(
      `/api/projections/${encodeURIComponent(currentBundle.projection_id)}/filter`,
      { method: "POST", body: JSON.stringify({ temporal_filter: temporalFilter }) },
    );
    renderGraph();
  };

  const parseList = (value) =>
    value
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);

  const submitRevision = async (event) => {
    event.preventDefault();
    if (!currentBundle) return;
    const action = byId("revision-action").value;
    const groups = byId("revision-mentions")
      .value.split("|")
      .map(parseList)
      .filter((item) => item.length > 0);
    const evidenceIds = parseList(byId("revision-evidence").value);
    const allMentions = groups.flat();
    const revisedStoryPoint = byId("revision-story-point").value;
    const revisedDiscourse = byId("revision-discourse").value;
    const revisedRevelation = byId("revision-revelation").value;
    const anchor = {
      evidence_ids: evidenceIds,
      mention_candidate_ids: allMentions,
      requested_semantic_signature: byId("revision-signature").value,
    };
    const payload = {
      before_projection_id: currentBundle.projection_id,
      action,
      anchors: [anchor],
      rationale: byId("revision-rationale").value,
      sequence,
      seed: 0,
      lens: action === "REFINE_CONTEXT" ? byId("revision-lens").value || null : null,
      story_scope:
        action === "REFINE_CONTEXT" && revisedStoryPoint !== ""
          ? { kind: "point", point: Number(revisedStoryPoint) }
          : null,
      spoiler_horizon:
        action === "REFINE_CONTEXT" && revisedDiscourse !== ""
          ? {
              horizon_id: `revision-horizon-${sequence}`,
              max_discourse_position: {
                passage_order: Number(revisedDiscourse),
                sentence_order: 0,
                token_order: 0,
              },
              max_revelation_position:
                revisedRevelation === ""
                  ? null
                  : { revelation_order: Number(revisedRevelation) },
              withheld_after_horizon: true,
            }
          : null,
      merge_split_operation:
        action === "REQUEST_MERGE_SPLIT" ? byId("revision-operation").value : null,
      grouped_mention_candidate_ids:
        action === "REQUEST_MERGE_SPLIT" ? groups : [],
    };
    revisionStatus.textContent = "Submitting…";
    try {
      const result = await api("/api/revisions", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      sequence += 1;
      revisionStatus.textContent = `Resolved in ${result.latency_seconds.toFixed(2)} s; replay ${result.replay.replay_hash_success ? "verified" : "failed"}.`;
      await initializeProjectionMenus(result.after_bundle.projection_id);
    } catch (error) {
      revisionStatus.textContent = error.message;
    }
  };

  const initializeProjectionMenus = async (selectedId = null) => {
    const projections = await api("/api/projections");
    projectionSelect.innerHTML = projections
      .map(
        (item) =>
          `<option value="${escapeHtml(item.projection_id)}">${escapeHtml(item.condition)} · ${escapeHtml(item.context_id)}</option>`,
      )
      .join("");
    compareSelect.innerHTML = `<option value="">No comparison</option>${projectionSelect.innerHTML}`;
    if (selectedId) projectionSelect.value = selectedId;
    if (projectionSelect.value) await loadProjection(projectionSelect.value);
  };

  const initialize = async () => {
    try {
      const health = await api("/api/health");
      cytoscapeAvailable = await window.storyProjectionCytoscapeReady;
      const verified = health.cytoscape.verified && cytoscapeAvailable;
      assetStatus.textContent = verified
        ? `Cytoscape.js ${health.cytoscape.version} verified locally`
        : "Cytoscape.js is not yet vendored and verified; see ui/README.md";
      assetStatus.classList.toggle("warning", !verified);
      await initializeProjectionMenus();
    } catch (error) {
      assetStatus.textContent = error.message;
      assetStatus.classList.add("error");
    }
  };

  projectionSelect.addEventListener("change", () => loadProjection(projectionSelect.value));
  compareSelect.addEventListener("change", () => loadProjection(projectionSelect.value));
  byId("apply-filter").addEventListener("click", () => applyFilter().catch(console.error));
  byId("clear-filter").addEventListener("click", () => {
    byId("story-point").value = "";
    byId("discourse-horizon").value = "";
    byId("revelation-horizon").value = "";
    loadProjection(projectionSelect.value).catch(console.error);
  });
  byId("revision-form").addEventListener("submit", submitRevision);
  initialize();
})();
