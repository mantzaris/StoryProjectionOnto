(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const projectionSelect = byId("projection-select");
  const compareSelect = byId("compare-select");
  const detailPanel = byId("detail-panel");
  const assetStatus = byId("asset-status");
  const semanticAssessmentStatus = byId("semantic-assessment-status");
  const revisionStatus = byId("revision-status");
  let cytoscapeAvailable = false;
  let cy = null;
  let currentBundle = null;
  let comparisonBundle = null;
  let currentDiff = null;
  let revisionSeedDecimal = null;
  let sequence = 1;
  const pageParameters = new URLSearchParams(window.location.search);
  const geometryCaptureMode = pageParameters.get("geometry_capture") === "1";
  const requestedProjectionId = pageParameters.get("projection_id");
  const frozenFontFamily = "system-ui,sans-serif";
  const frozenFontBasePx = 14;

  const roundedCoordinate = (value) => Math.round(Number(value) * 1000000) / 1000000;
  const codeUnitCompare = (left, right) => (left < right ? -1 : left > right ? 1 : 0);
  const coordinateKey = ({ x, y }) =>
    `${roundedCoordinate(x).toFixed(6)}:${roundedCoordinate(y).toFixed(6)}`;
  const stableCodeUnitHash = (value) => {
    let result = 2166136261;
    for (let index = 0; index < value.length; index += 1) {
      result = Math.imul(result ^ value.charCodeAt(index), 16777619) >>> 0;
    }
    return result;
  };
  const hubId = (assertionId) => `render-hub:${assertionId}`;
  const hubLabelId = (assertionId) => `render-label:${assertionId}:hub`;
  const roleLabelId = (assertionId, index) =>
    `render-label:${assertionId}:role:${index}`;

  const naryHubPositions = (state) => {
    const positions = new Map(
      state.positions.map((item) => [
        item.visualization_node_id,
        { x: item.x, y: item.y },
      ]),
    );
    const occupied = new Set([...positions.values()].map(coordinateKey));
    const visibleAssertions = new Set(state.visible_assertion_ids);
    const result = new Map();
    const assertions = [...state.assertions].sort((left, right) =>
      codeUnitCompare(left.visualization_assertion_id, right.visualization_assertion_id),
    );
    for (const assertion of assertions) {
      if (!visibleAssertions.has(assertion.visualization_assertion_id) || !assertion.roles.length) {
        continue;
      }
      const rolePositions = assertion.roles.map((role) => positions.get(role.object_id));
      if (rolePositions.some((item) => !item)) {
        throw new Error("n-ary renderer hub references an unpositioned role");
      }
      const centerX = rolePositions.reduce((total, item) => total + item.x, 0) /
        rolePositions.length;
      const centerY = rolePositions.reduce((total, item) => total + item.y, 0) /
        rolePositions.length;
      const identifierHash = stableCodeUnitHash(assertion.visualization_assertion_id);
      let offsetX = identifierHash % 49 - 24;
      const offsetY = Math.floor(identifierHash / 49) % 49 - 24;
      if (offsetX === 0 && offsetY === 0) offsetX = 25;
      let attempt = 0;
      while (true) {
        const candidate = {
          x: roundedCoordinate(centerX + offsetX + attempt * 53),
          y: roundedCoordinate(centerY + offsetY + attempt * 47),
        };
        const key = coordinateKey(candidate);
        if (key !== "0.000000:0.000000" && !occupied.has(key)) {
          occupied.add(key);
          result.set(assertion.visualization_assertion_id, candidate);
          break;
        }
        attempt += 1;
      }
    }
    return result;
  };

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
    if (time.kind === "partial_order") {
      return time.partial_order
        .map((item) => `${item.left_id} ${item.relation} ${item.right_id}`)
        .join("; ");
    }
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
        <dt>Release class</dt><dd>${escapeHtml(item.release_class)}</dd>
            </dl>
            ${item.public_text ? `<p class="evidence-text">${escapeHtml(item.public_text)}</p>` : ""}
          `);
        } catch (error) {
          renderDetails(`<p class="error">${escapeHtml(error.message)}</p>`);
        }
      });
    });
  };

  const nodeDetail = (id, bundle = currentBundle) => {
    const node = bundle.state.nodes.find((item) => item.visualization_node_id === id);
    const detail = bundle.node_details.find((item) => item.visualization_node_id === id);
    if (!node || !detail) return;
    const descriptionLabel =
      detail.description_support_status === "supported"
        ? "Projection description (verified)"
        : "Projection-authored node description (not verified)";
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
        <dt>Description support</dt><dd>${escapeHtml(detail.description_support_status)}</dd>
        <dt>Stable anchor</dt><dd><code>${escapeHtml(detail.stable_anchor_id)}</code></dd>
      </dl>
      <p><strong>${descriptionLabel}:</strong> ${escapeHtml(node.description)}</p>
      <p><strong>Evidence (${node.evidence_badge.count})</strong><br />${evidenceButtons(node.evidence_badge.evidence_ids)}</p>
    `);
    bindEvidenceButtons();
  };

  const assertionDetail = (id, bundle = currentBundle) => {
    const assertion = bundle.state.assertions.find(
      (item) => item.visualization_assertion_id === id,
    );
    const detail = bundle.assertion_details.find(
      (item) => item.visualization_assertion_id === id,
    );
    if (!assertion || !detail) return;
    const evidenceById = new Map(bundle.evidence_metadata.map((item) => [item.evidence_id, item]));
    const provenance = assertion.provenance
      .map((item) => {
        const metadata = evidenceById.get(item.evidence_id);
        const locator = metadata ? metadata.locator : item.locator;
        return `${item.evidence_id}: ${item.extraction_method} @ ${locator}`;
      })
      .join("; ");
    const roles = assertion.roles
      .map((item) => `${item.role} → ${item.object_id}`)
      .join("; ");
    const holderNode = bundle.state.nodes.find(
      (item) => item.visualization_node_id === assertion.epistemic_holder_id,
    );
    const holderLabel = holderNode
      ? `${holderNode.contextual_label} (${assertion.epistemic_holder_id})`
      : assertion.epistemic_holder_id || "none";
    const descriptionEvidenceLabel =
      detail.description_support_status === "supported"
        ? "Verified description evidence"
        : detail.description_support_status === "pending"
          ? "Projection-claimed description evidence (not verified)"
          : "Description evidence";
    const descriptionEvidence =
      detail.description_support_status === "supported" ||
      detail.description_support_status === "pending"
        ? detail.why_matters_evidence_ids.length
          ? evidenceButtons(detail.why_matters_evidence_ids)
          : "none"
        : "none verified";
    renderDetails(`
      <h3>${escapeHtml(assertion.contextual_label)}</h3>
      <dl>
        <dt>Relation definition</dt><dd>${escapeHtml(detail.predicate_definition)}</dd>
        <dt>Normalized relation</dt><dd><code>${escapeHtml(assertion.predicate_id)}</code></dd>
        <dt>Direction</dt><dd>${escapeHtml(assertion.direction)}</dd>
        <dt>Event roles</dt><dd>${escapeHtml(roles || "binary assertion")}</dd>
        <dt>Story time</dt><dd>${escapeHtml(temporalLabel(assertion.temporal_scope.story_time))}</dd>
        <dt>Validity</dt><dd>${escapeHtml(temporalLabel(assertion.temporal_scope.validity_time))}</dd>
        <dt>Discourse</dt><dd>${assertion.temporal_scope.discourse_position.passage_order}</dd>
        <dt>Revelation</dt><dd>${assertion.temporal_scope.revelation_position.revelation_order}</dd>
        <dt>Commitment</dt><dd>${escapeHtml(detail.narrative_commitment)}</dd>
        <dt>Holder / attitude</dt><dd>${escapeHtml(holderLabel)} / ${escapeHtml(assertion.epistemic_attitude || "none")}</dd>
        <dt>Holder-relative time</dt><dd>${escapeHtml(temporalLabel(detail.holder_relative_time))}</dd>
        <dt>Proposition content</dt><dd>${escapeHtml(detail.proposition_content_id || "none")}</dd>
        <dt>Uncertainty</dt><dd>${escapeHtml(assertion.uncertainty)}</dd>
        <dt>Confidence</dt><dd>${Number(assertion.confidence).toFixed(2)}</dd>
        <dt>Contextual relevance</dt><dd>${Number(detail.contextual_relevance).toFixed(2)}</dd>
        <dt>Assertion support</dt><dd>${escapeHtml(detail.assertion_support_status)}</dd>
        <dt>Description support</dt><dd>${escapeHtml(detail.description_support_status)}</dd>
        <dt>Provenance</dt><dd>${escapeHtml(provenance)}</dd>
      </dl>
      <p><strong>${detail.description_support_status === "supported" ? "Why it matters (verified):" : "Projection-authored why-it-matters (not verified):"}</strong> ${escapeHtml(assertion.why_matters)}</p>
      <p><strong>${descriptionEvidenceLabel}:</strong><br />${descriptionEvidence}</p>
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
    const hubPositions = naryHubPositions(bundle.state);
    const details = new Map(
      bundle.node_details.map((item) => [item.visualization_node_id, item]),
    );
    const assertionDetails = new Map(
      bundle.assertion_details.map((item) => [item.visualization_assertion_id, item]),
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
          label: `${node.contextual_label}\n${detail.contextual_type_label} · ${node.contextual_role}\nt=${temporalLabel(node.temporal_state)} · u=${node.uncertainty} · c=${Number(node.confidence).toFixed(2)} · E${node.evidence_badge.count}`,
          subtitle: `${detail.contextual_type_label} · ${node.contextual_role}`,
          confidence: node.confidence,
          evidenceCount: node.evidence_badge.count,
          labelComponentId: node.projection_object_id,
          labelSemanticId: node.projection_object_id,
        },
        position: positions.get(node.visualization_node_id),
        classes: `${detail.object_kind} semantic-${detail.description_support_status} ${classes.get(node.visualization_node_id) || ""}`,
      });
    }
    for (const assertion of bundle.state.assertions) {
      if (!visibleAssertions.has(assertion.visualization_assertion_id)) continue;
      const assertionSupport = assertionDetails.get(
        assertion.visualization_assertion_id,
      )?.assertion_support_status || "pending";
      const edgeClass = `${assertion.epistemic_holder_id ? "epistemic" : ""} semantic-${assertionSupport} ${classes.get(assertion.visualization_assertion_id) || ""}`;
      const epistemicStatus = assertion.epistemic_holder_id
        ? `${assertion.epistemic_attitude}@${assertion.epistemic_holder_id}`
        : `${assertion.uncertainty} · c=${Number(assertion.confidence).toFixed(2)}`;
      if (assertion.source_visualization_node_id) {
        elements.push({
          group: "edges",
          data: {
            id: assertion.visualization_assertion_id,
            source: assertion.source_visualization_node_id,
            target: assertion.target_visualization_node_id,
            label: `${assertion.contextual_label} · ${temporalLabel(assertion.temporal_scope.validity_time)} · ${epistemicStatus} · E${assertion.evidence_badge.count}`,
            confidence: assertion.confidence,
            evidenceCount: assertion.evidence_badge.count,
            semanticId: assertion.visualization_assertion_id,
            labelComponentId: assertion.projection_assertion_id,
            labelSemanticId: assertion.projection_assertion_id,
          },
          classes: edgeClass,
        });
      } else {
        const assertionHubId = hubId(assertion.visualization_assertion_id);
        elements.push({
          group: "nodes",
          data: {
            id: assertionHubId,
            semanticId: assertion.visualization_assertion_id,
            assertionId: assertion.visualization_assertion_id,
            label: assertion.contextual_label,
            labelComponentId: hubLabelId(assertion.visualization_assertion_id),
            labelSemanticId: assertion.projection_assertion_id,
            rendererOnly: true,
          },
          position: hubPositions.get(assertion.visualization_assertion_id),
          classes: `assertion-hub ${edgeClass}`,
        });
        assertion.roles.forEach((role, index) => {
          elements.push({
            group: "edges",
            data: {
              id: `render-role:${assertion.visualization_assertion_id}:${index}`,
              assertionId: assertion.visualization_assertion_id,
              semanticId: assertion.visualization_assertion_id,
              source: assertionHubId,
              target: role.object_id,
              label: role.role,
              labelComponentId: roleLabelId(assertion.visualization_assertion_id, index),
              labelSemanticId: assertion.projection_assertion_id,
            },
            classes: `role-edge ${edgeClass}`,
          });
        });
      }
    }
    if (comparisonBundle && currentDiff) {
      const ghostNodeIds = new Map();
      const ghostAssertionIds = new Map();
      for (const change of currentDiff.changes) {
        if (!["removed", "merged", "split"].includes(change.kind)) continue;
        const target = change.object_kind === "node" ? ghostNodeIds : ghostAssertionIds;
        for (const id of change.before_visualization_ids) target.set(id, change.kind);
      }
      const currentNodeIds = new Set(
        elements.filter((item) => item.group === "nodes").map((item) => item.data.id),
      );
      const comparisonPositions = new Map(
        comparisonBundle.state.positions.map((item) => [
          item.visualization_node_id,
          { x: item.x, y: item.y },
        ]),
      );
      const comparisonHubPositions = naryHubPositions(comparisonBundle.state);
      const comparisonDetails = new Map(
        comparisonBundle.node_details.map((item) => [item.visualization_node_id, item]),
      );
      for (const node of comparisonBundle.state.nodes) {
        if (!ghostNodeIds.has(node.visualization_node_id)) continue;
        const ghostId = `before:${node.visualization_node_id}`;
        const detail = comparisonDetails.get(node.visualization_node_id);
        elements.push({
          group: "nodes",
          data: {
            id: ghostId,
            semanticId: node.visualization_node_id,
            sourceBundle: "comparison",
            label: `Before: ${node.contextual_label}\n${detail.contextual_type_label} · ${node.contextual_role}`,
          },
          position: comparisonPositions.get(node.visualization_node_id),
          classes: `${detail.object_kind} before-state ${ghostNodeIds.get(node.visualization_node_id)}`,
        });
        currentNodeIds.add(ghostId);
      }
      const comparisonEndpoint = (id) => {
        if (ghostNodeIds.has(id)) return `before:${id}`;
        return currentNodeIds.has(id) ? id : null;
      };
      for (const assertion of comparisonBundle.state.assertions) {
        if (!ghostAssertionIds.has(assertion.visualization_assertion_id)) continue;
        const ghostId = `before:${assertion.visualization_assertion_id}`;
        const classes = `before-state ${ghostAssertionIds.get(assertion.visualization_assertion_id)}`;
        if (assertion.source_visualization_node_id) {
          const source = comparisonEndpoint(assertion.source_visualization_node_id);
          const target = comparisonEndpoint(assertion.target_visualization_node_id);
          if (!source || !target) continue;
          elements.push({
            group: "edges",
            data: {
              id: ghostId,
              semanticId: assertion.visualization_assertion_id,
              sourceBundle: "comparison",
              source,
              target,
              label: `Before: ${assertion.contextual_label}`,
            },
            classes,
          });
        } else {
          const endpoints = assertion.roles.map((role) => comparisonEndpoint(role.object_id));
          if (endpoints.some((item) => !item)) continue;
          const hubId = `before:render-hub:${assertion.visualization_assertion_id}`;
          elements.push({
            group: "nodes",
            data: {
              id: hubId,
              semanticId: assertion.visualization_assertion_id,
              sourceBundle: "comparison",
              label: `Before: ${assertion.contextual_label}`,
              rendererOnly: true,
            },
            position: comparisonHubPositions.get(assertion.visualization_assertion_id),
            classes: `assertion-hub ${classes}`,
          });
          assertion.roles.forEach((role, index) => {
            elements.push({
              group: "edges",
              data: {
                id: `${ghostId}:role:${index}`,
                semanticId: assertion.visualization_assertion_id,
                sourceBundle: "comparison",
                source: hubId,
                target: endpoints[index],
                label: role.role,
              },
              classes: `role-edge ${classes}`,
            });
          });
        }
      }
    }
    return elements;
  };

  const renderGraph = () => {
    if (!currentBundle || !cytoscapeAvailable) return;
    if (cy) cy.destroy();
    const graphContainer = byId("graph");
    if (geometryCaptureMode) {
      const { width, height } = currentBundle.state.viewport;
      graphContainer.style.position = "fixed";
      graphContainer.style.inset = "0 auto auto 0";
      graphContainer.style.width = `${width}px`;
      graphContainer.style.height = `${height}px`;
      graphContainer.style.minHeight = `${height}px`;
      graphContainer.style.zIndex = "10000";
      graphContainer.style.background = "#081017";
    }
    const viewport = currentBundle.state.viewport;
    const fixedPan = {
      x: viewport.width / 2 - viewport.center_x * viewport.zoom,
      y: viewport.height / 2 - viewport.center_y * viewport.zoom,
    };
    cy = window.cytoscape({
      container: graphContainer,
      elements: graphElements(currentBundle),
      layout: {
        name: "preset",
        fit: !geometryCaptureMode,
        padding: geometryCaptureMode ? 0 : 48,
      },
      zoom: geometryCaptureMode ? viewport.zoom : undefined,
      pan: geometryCaptureMode ? fixedPan : undefined,
      pixelRatio: geometryCaptureMode ? 1 : "auto",
      minZoom: 0.2,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            label: "data(label)",
            "font-family": frozenFontFamily,
            "font-size": frozenFontBasePx,
            "font-style": "normal",
            "font-weight": "normal",
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
          style: {
            shape: "round-rectangle",
            width: 20,
            height: 20,
            "font-family": frozenFontFamily,
            "font-size": frozenFontBasePx,
            "font-style": "normal",
            "font-weight": "normal",
          },
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
            "font-family": frozenFontFamily,
            "font-size": frozenFontBasePx,
            "font-style": "normal",
            "font-weight": "normal",
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
        { selector: ".before-state", style: { opacity: 0.38, "line-style": "dotted", "border-style": "dotted" } },
        { selector: ":selected", style: { "overlay-color": "#ffffff", "overlay-opacity": 0.12 } },
      ],
    });
    if (geometryCaptureMode) {
      cy.zoom(viewport.zoom);
      cy.pan(fixedPan);
      cy.resize();
    }
    cy.on("tap", "node", (event) => {
      const id = event.target.id();
      const bundle = event.target.data("sourceBundle") === "comparison" ? comparisonBundle : currentBundle;
      const semanticId = event.target.data("semanticId") || id;
      if (!event.target.data("rendererOnly")) nodeDetail(semanticId, bundle);
      else assertionDetail(semanticId, bundle);
    });
    cy.on("tap", "edge", (event) => {
      const bundle = event.target.data("sourceBundle") === "comparison" ? comparisonBundle : currentBundle;
      assertionDetail(
        event.target.data("semanticId") || event.target.data("assertionId") || event.target.id(),
        bundle,
      );
    });
  };

  const renderedLabelRectangle = (element) => {
    const box = element.renderedBoundingBox({
      includeNodes: false,
      includeEdges: false,
      includeLabels: true,
      includeOverlays: false,
      includeShadows: false,
    });
    return {
      left: roundedCoordinate(box.x1),
      top: roundedCoordinate(box.y1),
      right: roundedCoordinate(box.x2),
      bottom: roundedCoordinate(box.y2),
    };
  };

  const canonicalFontFamily = (value) =>
    String(value).toLowerCase().replaceAll(" ", "").replaceAll('"', "").replaceAll("'", "");

  const captureTypographyStyles = () => {
    const probe = cy.add([
      {
        group: "nodes",
        data: { id: "render-probe:node", label: "probe" },
        position: { x: -10000, y: -10000 },
      },
      {
        group: "nodes",
        data: { id: "render-probe:hub", label: "probe" },
        position: { x: -10020, y: -10020 },
        classes: "assertion-hub",
      },
      {
        group: "edges",
        data: {
          id: "render-probe:edge",
          source: "render-probe:node",
          target: "render-probe:hub",
          label: "probe",
        },
      },
    ]);
    const effectiveStyle = (identifier, elementKind) => {
      const element = cy.$id(identifier);
      return {
        element_kind: elementKind,
        font_family: canonicalFontFamily(element.style("font-family")),
        font_size_px: roundedCoordinate(parseFloat(element.style("font-size"))),
        font_style: element.style("font-style"),
        font_weight: element.style("font-weight"),
      };
    };
    const styles = [
      effectiveStyle("render-probe:edge", "edge"),
      effectiveStyle("render-probe:hub", "hub"),
      effectiveStyle("render-probe:node", "node"),
    ];
    probe.remove();
    return styles;
  };

  const nextPaint = () =>
    new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));

  window.storyProjectionCaptureGeometry = async () => {
    if (!geometryCaptureMode || !cy || !currentBundle) {
      throw new Error("geometry capture is available only in the dedicated capture mode");
    }
    if (document.fonts?.ready) await document.fonts.ready;
    await nextPaint();
    cy.resize();
    const typographyStyles = captureTypographyStyles();
    await nextPaint();
    const { state } = currentBundle;
    const nodeByVisualizationId = new Map(
      state.nodes.map((item) => [item.visualization_node_id, item]),
    );
    const assertionByVisualizationId = new Map(
      state.assertions.map((item) => [item.visualization_assertion_id, item]),
    );
    const positions = [];
    const rendererOnlyPositions = [];
    const labelRectangles = [];
    const labelSemanticIds = [];
    for (const visualizationId of state.visible_node_ids) {
      const semantic = nodeByVisualizationId.get(visualizationId);
      const element = cy.$id(visualizationId);
      if (!semantic || element.length !== 1 || !element.visible()) {
        throw new Error(`visible semantic node is absent from renderer: ${visualizationId}`);
      }
      const position = element.renderedPosition();
      positions.push([
        semantic.projection_object_id,
        { x: roundedCoordinate(position.x), y: roundedCoordinate(position.y) },
      ]);
      labelRectangles.push({
        label_id: element.data("labelComponentId"),
        ...renderedLabelRectangle(element),
      });
      labelSemanticIds.push([
        element.data("labelComponentId"),
        element.data("labelSemanticId"),
      ]);
    }
    const interactiveAssertions = [];
    for (const visualizationId of state.visible_assertion_ids) {
      const semantic = assertionByVisualizationId.get(visualizationId);
      if (!semantic) throw new Error(`unknown visual assertion: ${visualizationId}`);
      const rendered = cy.elements().filter((element) => {
        const semanticId =
          element.data("semanticId") || element.data("assertionId") || element.id();
        return semanticId === visualizationId && element.visible();
      });
      if (rendered.length === 0) {
        throw new Error(`visible semantic assertion is absent from renderer: ${visualizationId}`);
      }
      const labeledElements = Array.from(rendered)
        .filter((element) => Boolean(element.data("label")))
        .sort((left, right) =>
          codeUnitCompare(left.data("labelComponentId"), right.data("labelComponentId")),
        );
      if (labeledElements.length === 0) {
        throw new Error(`visible semantic assertion has no rendered label: ${visualizationId}`);
      }
      for (const element of labeledElements) {
        const componentId = element.data("labelComponentId");
        const semanticId = element.data("labelSemanticId");
        if (!componentId || semanticId !== semantic.projection_assertion_id) {
          throw new Error(`renderer label binding drifted for: ${visualizationId}`);
        }
        labelRectangles.push({
          label_id: componentId,
          ...renderedLabelRectangle(element),
        });
        labelSemanticIds.push([componentId, semanticId]);
      }
      if (semantic.roles.length) {
        const element = cy.$id(hubId(visualizationId));
        if (element.length !== 1 || !element.visible()) {
          throw new Error(`visible n-ary hub is absent from renderer: ${visualizationId}`);
        }
        const position = element.renderedPosition();
        rendererOnlyPositions.push([
          element.id(),
          { x: roundedCoordinate(position.x), y: roundedCoordinate(position.y) },
        ]);
      }
      interactiveAssertions.push(semantic.projection_assertion_id);
    }
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    context.font = `${frozenFontBasePx}px ${frozenFontFamily}`;
    const fontProbeCss = context.font;
    const fontProbeWidths = ["StoryProjectionOnto", "belief@holder", "WMWMiiii"]
      .map((value) => roundedCoordinate(context.measureText(value).width));
    const visibleSemanticIds = [
      ...state.visible_node_ids.map(
        (id) => nodeByVisualizationId.get(id).projection_object_id,
      ),
      ...state.visible_assertion_ids.map(
        (id) => assertionByVisualizationId.get(id).projection_assertion_id,
      ),
    ].sort(codeUnitCompare);
    return {
      capture_protocol: "cytoscape-browser-geometry-v1",
      projection_hash: currentBundle.projection_hash,
      visualization_bundle_hash: currentBundle.content_hash,
      visualization_state_hash: state.content_hash,
      layout_config_hash: state.layout_config_hash,
      style_config_hash: state.style_config_hash,
      font_config_hash: state.font_config_hash,
      viewport: state.viewport,
      labels_visible: state.labels_visible,
      temporal_filter_hash: state.temporal_filter.content_hash,
      positions: positions.sort((left, right) => codeUnitCompare(left[0], right[0])),
      renderer_only_positions: rendererOnlyPositions.sort((left, right) =>
        codeUnitCompare(left[0], right[0]),
      ),
      label_rectangles: labelRectangles.sort((left, right) =>
        codeUnitCompare(left.label_id, right.label_id),
      ),
      label_semantic_ids: labelSemanticIds.sort((left, right) =>
        codeUnitCompare(left[0], right[0]),
      ),
      visible_semantic_ids: visibleSemanticIds,
      interactive_assertion_ids: interactiveAssertions.sort(codeUnitCompare),
      renderer_runtime: {
        cytoscape_version: window.cytoscape.version,
        user_agent: navigator.userAgent,
        platform: navigator.platform || "unknown",
        device_pixel_ratio: window.devicePixelRatio,
        container_width: byId("graph").clientWidth,
        container_height: byId("graph").clientHeight,
        document_fonts_status: document.fonts?.status || "unavailable",
        font_probe_available: document.fonts?.check(fontProbeCss) || false,
        font_probe_css: fontProbeCss,
        font_probe_widths: fontProbeWidths,
        typography_styles: typographyStyles,
      },
    };
  };

  const loadProjection = async (projectionId) => {
    currentBundle = await api(`/api/projections/${encodeURIComponent(projectionId)}`);
    if (
      ![
        "pending_scorer_or_reviewer",
        "verified_scorer_or_reviewer",
      ].includes(currentBundle.semantic_assessment_status)
    ) {
      throw new Error("projection has an unrecognized or unbound semantic assessment status");
    }
    if (currentBundle.semantic_assessment_status === "pending_scorer_or_reviewer") {
      semanticAssessmentStatus.textContent =
        "Semantic support: pending scorer or reviewer assessment; graph shows structurally accepted output in its declared content scope";
      semanticAssessmentStatus.classList.add("warning");
    } else {
      if (!currentBundle.semantic_overlay) {
        throw new Error("verified semantic status lacks its bound assessment overlay");
      }
      const source = currentBundle.semantic_overlay.source_kind;
      const mode = currentBundle.semantic_display_mode;
      semanticAssessmentStatus.textContent =
        `Semantic support: verified ${source} overlay; ${mode === "supported_only" ? "unsupported assertions are omitted and unverified descriptions are withheld" : "all selected output is retained and support status is disclosed"}`;
      semanticAssessmentStatus.classList.remove("warning");
    }
    comparisonBundle = null;
    currentDiff = null;
    const compareId = compareSelect.value;
    if (compareId && compareId !== projectionId) {
      comparisonBundle = await api(`/api/projections/${encodeURIComponent(compareId)}`);
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
                sentence_order: 2147483647,
                token_order: 2147483647,
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
    if (revisionSeedDecimal === null) {
      revisionStatus.textContent =
        "No frozen feedback seed is configured; no regeneration was started.";
      return;
    }
    const beforeProjectionId = currentBundle.projection_id;
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
      // Keep the 63-bit registered seed as decimal text to avoid IEEE-754 rounding.
      seed: revisionSeedDecimal,
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
                sentence_order: 2147483647,
                token_order: 2147483647,
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
      if (result.resolution.status === "capability_limited") {
        revisionStatus.textContent = `Capability limited for ${currentBundle.condition}; no semantic output was claimed.`;
      } else if (!result.after_bundle) {
        revisionStatus.textContent = `Revision ${result.resolution.status}; no valid output was retained.`;
      } else {
        revisionStatus.textContent = `Resolved in ${result.latency_seconds.toFixed(2)} s; replay ${result.replay.replay_hash_success ? "verified" : "failed"}.`;
        await initializeProjectionMenus(
          result.after_bundle.projection_id,
          beforeProjectionId,
        );
      }
    } catch (error) {
      revisionStatus.textContent = error.message;
    }
  };

  const initializeProjectionMenus = async (selectedId = null, compareId = null) => {
    const projections = await api("/api/projections");
    projectionSelect.innerHTML = projections
      .map(
        (item) =>
          `<option value="${escapeHtml(item.projection_id)}">${escapeHtml(item.condition)} · ${escapeHtml(item.context_id)}</option>`,
      )
      .join("");
    compareSelect.innerHTML = `<option value="">No comparison</option>${projectionSelect.innerHTML}`;
    if (selectedId) projectionSelect.value = selectedId;
    if (compareId) compareSelect.value = compareId;
    if (projectionSelect.value) await loadProjection(projectionSelect.value);
  };

  const initialize = async () => {
    try {
      const health = await api("/api/health");
      revisionSeedDecimal = health.revision_seed_decimal;
      cytoscapeAvailable = await window.storyProjectionCytoscapeReady;
      const verified = health.cytoscape.verified && cytoscapeAvailable;
      assetStatus.textContent = verified
        ? `Cytoscape.js ${health.cytoscape.version} verified locally`
        : "Cytoscape.js is not yet vendored and verified; see ui/README.md";
      assetStatus.classList.toggle("warning", !verified);
      await initializeProjectionMenus(requestedProjectionId);
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
