/* The real library is deliberately not impersonated by a stub. */
window.storyProjectionCytoscapeReady = new Promise((resolve) => {
  const script = document.createElement("script");
  script.src = "/static/cytoscape.min.js";
  script.onload = () => resolve(typeof window.cytoscape === "function");
  script.onerror = () => resolve(false);
  document.head.appendChild(script);
});
