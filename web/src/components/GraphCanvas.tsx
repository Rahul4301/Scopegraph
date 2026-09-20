import cytoscape, { type Core, type ElementDefinition, type StylesheetJson } from "cytoscape";
import { useEffect, useRef } from "react";

import type { GraphSubgraph } from "../types";

interface GraphCanvasProps {
  graph: GraphSubgraph | null;
  selectedNodeId: string | null;
  retrievedIds: Set<string>;
  onSelectNode: (nodeId: string) => void;
}

const styles: StylesheetJson = [
  {
    selector: "node",
    style: {
      label: "data(displayLabel)",
      color: "#dce7e5",
      "font-size": 10,
      "font-family": "IBM Plex Mono, ui-monospace, monospace",
      "text-wrap": "ellipsis",
      "text-max-width": "120px",
      "text-valign": "bottom",
      "text-margin-y": 8,
      "background-color": "#52756f",
      "border-color": "#8db7af",
      "border-width": 1.5,
      width: 34,
      height: 34,
    },
  },
  {
    selector: 'node[nodeType = "scope"]',
    style: {
      shape: "round-rectangle",
      width: 58,
      height: 30,
      "background-color": "#213b38",
      "border-color": "#74a89e",
    },
  },
  {
    selector: 'node[nodeType = "source_message"]',
    style: {
      shape: "diamond",
      width: 26,
      height: 26,
      "background-color": "#655a45",
      "border-color": "#c6a86c",
    },
  },
  {
    selector: 'node[scopeLevel = "session"]',
    style: { "background-color": "#4e6f91", "border-color": "#8fb7dc" },
  },
  {
    selector: 'node[scopeLevel = "global"]',
    style: { "background-color": "#6f4f7e", "border-color": "#c19bcf" },
  },
  {
    selector: 'node[status = "superseded"]',
    style: { "border-style": "dashed", opacity: 0.66 },
  },
  {
    selector: 'node[status = "archived"]',
    style: { "border-style": "dotted", opacity: 0.58 },
  },
  {
    selector: 'node[status = "tombstoned"]',
    style: {
      "background-color": "#392c2d",
      "border-color": "#b57171",
      "border-style": "double",
      opacity: 0.54,
    },
  },
  {
    selector: 'node[status = "needs_review"]',
    style: {
      "background-color": "#78562d",
      "border-color": "#f0b765",
      "border-width": 3,
    },
  },
  {
    selector: "node.retrieved",
    style: { "border-color": "#f1d37b", "border-width": 4, "overlay-opacity": 0 },
  },
  {
    selector: "node:selected",
    style: { "border-color": "#ffffff", "border-width": 4, "overlay-opacity": 0 },
  },
  {
    selector: "edge",
    style: {
      width: 1.2,
      "line-color": "#47605c",
      "target-arrow-color": "#47605c",
      "target-arrow-shape": "triangle",
      "curve-style": "bezier",
      label: "data(displayLabel)",
      color: "#81918e",
      "font-size": 8,
      "text-background-color": "#101817",
      "text-background-opacity": 0.82,
      "text-background-padding": "2px",
    },
  },
  {
    selector: 'edge[relation = "SUPPORTS"]',
    style: { "line-color": "#83a96b", "target-arrow-color": "#83a96b" },
  },
  {
    selector: 'edge[relation = "CONTRADICTS"]',
    style: { "line-color": "#bd6c69", "target-arrow-color": "#bd6c69", "line-style": "dashed" },
  },
  {
    selector: 'edge[relation = "DERIVED_FROM"]',
    style: { "line-color": "#8a7958", "target-arrow-color": "#8a7958", "line-style": "dotted" },
  },
];

export function GraphCanvas({ graph, selectedNodeId, retrievedIds, onSelectNode }: GraphCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current || !graph) return;
    const elements: ElementDefinition[] = [
      ...graph.nodes.map((node) => {
        const status = String(node.data.status ?? "");
        return {
          data: {
            id: node.id,
            nodeType: node.node_type,
            scopeLevel: node.data.scope_level ?? "",
            status,
            displayLabel: status ? `[${status}] ${node.label}` : node.label,
          },
        };
      }),
      ...graph.edges.map((edge) => ({
        data: {
          id: edge.id,
          source: edge.source,
          target: edge.target,
          relation: edge.relation,
          displayLabel: edge.kind ?? edge.relation,
        },
      })),
    ];
    cyRef.current?.destroy();
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: styles,
      minZoom: 0.22,
      maxZoom: 2.4,
      wheelSensitivity: 0.18,
      layout: {
        name: "cose",
        animate: false,
        nodeRepulsion: () => 7200,
        idealEdgeLength: () => 95,
        gravity: 0.24,
        padding: 28,
      },
    });
    cy.on("tap", "node", (event) => onSelectNode(event.target.id()));
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph, onSelectNode]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().unselect();
    if (selectedNodeId) cy.getElementById(selectedNodeId).select();
  }, [selectedNodeId]);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.nodes().removeClass("retrieved");
    retrievedIds.forEach((id) => cy.getElementById(id).addClass("retrieved"));
  }, [retrievedIds]);

  return (
    <div className="graph-stage">
      <div ref={containerRef} className="graph-canvas" aria-label="Interactive memory graph" />
      {!graph?.nodes.length && (
        <div className="graph-empty">
          <span className="graph-empty__mark">⌁</span>
          <strong>No nodes in this view</strong>
          <p>Choose another scope or include inactive memories.</p>
        </div>
      )}
      <div className="graph-controls" aria-label="Graph controls">
        <button type="button" onClick={() => cyRef.current?.fit(undefined, 36)}>
          Fit
        </button>
        <button
          type="button"
          onClick={() =>
            cyRef.current?.layout({ name: "cose", animate: true, animationDuration: 350 }).run()
          }
        >
          Arrange
        </button>
      </div>
    </div>
  );
}
