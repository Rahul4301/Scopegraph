"""Export the local Neo4j graph as portable JSON."""

import argparse
import asyncio
import json
from pathlib import Path

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient


async def _export() -> dict[str, object]:
    client = Neo4jClient(get_settings())
    try:
        if not await client.health():
            raise SystemExit("Neo4j is unavailable; start it with `make neo4j-up`")
        rows = await client.execute_read(
            """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(target)
            RETURN collect(DISTINCT {
                       id: n.id, labels: labels(n), properties: properties(n)
                   }) AS nodes,
                   collect(DISTINCT CASE WHEN r IS NULL THEN NULL ELSE {
                       source_id: n.id, target_id: target.id, type: type(r),
                       properties: properties(r)
                   } END) AS edges
            """
        )
        payload = rows[0] if rows else {"nodes": [], "edges": []}
        payload["edges"] = [edge for edge in payload["edges"] if edge is not None]
        return payload
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/graph.json"))
    args = parser.parse_args()
    payload = asyncio.run(_export())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
