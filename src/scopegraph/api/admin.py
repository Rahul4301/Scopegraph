from typing import Annotated, Literal
from xml.sax.saxutils import escape, quoteattr

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from scopegraph.api.dependencies import get_repository
from scopegraph.graph.repository import Neo4jMemoryRepository
from scopegraph.models.graph import GraphSubgraph
from scopegraph.models.retrieval import MemoryStats

router = APIRouter(tags=["graph"])
Repository = Annotated[Neo4jMemoryRepository, Depends(get_repository)]


@router.get("/graph/subgraph", response_model=GraphSubgraph)
async def get_subgraph(
    repository: Repository,
    scope_id: Annotated[str | None, Query()] = None,
    memory_id: Annotated[str | None, Query()] = None,
    include_inactive: Annotated[bool, Query()] = True,
    include_sources: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> GraphSubgraph:
    if memory_id is not None and await repository.get_memory(memory_id) is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    if scope_id is not None and await repository.get_scope(scope_id) is None:
        raise HTTPException(status_code=404, detail="Scope not found")
    return await repository.get_subgraph(
        scope_id=scope_id,
        memory_id=memory_id,
        include_inactive=include_inactive,
        include_sources=include_sources,
        limit=limit,
    )


@router.get("/graph/export", response_model=None)
async def export_graph(
    repository: Repository,
    format: Annotated[Literal["json", "graphml"], Query()] = "json",
    scope_id: Annotated[str | None, Query()] = None,
    include_inactive: Annotated[bool, Query()] = True,
) -> Response:
    graph = await repository.get_subgraph(
        scope_id=scope_id,
        include_inactive=include_inactive,
        include_sources=True,
        limit=500,
    )
    if format == "json":
        return JSONResponse(graph.model_dump(mode="json"))
    return Response(
        content=_graphml(graph),
        media_type="application/graphml+xml",
        headers={"Content-Disposition": 'attachment; filename="scopegraph.graphml"'},
    )


@router.get("/stats", response_model=MemoryStats)
async def stats(repository: Repository) -> MemoryStats:
    return await repository.stats("scopegraph")


def _graphml(graph: GraphSubgraph) -> str:
    nodes = "".join(
        f"<node id={quoteattr(node.id)}><data key=\"type\">{node.node_type}</data>"
        f'<data key="label">{escape(node.label)}</data></node>'
        for node in graph.nodes
    )
    edges = "".join(
        f"<edge id={quoteattr(edge.id)} source={quoteattr(edge.source)} "
        f"target={quoteattr(edge.target)}><data key=\"relation\">"
        f"{escape(edge.relation)}</data></edge>"
        for edge in graph.edges
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">'
        '<key id="type" for="node" attr.name="type" attr.type="string"/>'
        '<key id="label" for="node" attr.name="label" attr.type="string"/>'
        '<key id="relation" for="edge" attr.name="relation" attr.type="string"/>'
        f'<graph id="scopegraph" edgedefault="directed">{nodes}{edges}</graph></graphml>'
    )
