"""Run all idempotent Neo4j schema migrations."""

import asyncio

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.schema import ensure_schema


async def main() -> None:
    client = Neo4jClient(get_settings())
    try:
        if not await client.health():
            raise SystemExit("Neo4j is unavailable; start it with `make neo4j-up`")
        await ensure_schema(client)
        print("Neo4j migrations are current.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
