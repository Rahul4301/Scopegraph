import asyncio

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient
from scopegraph.graph.schema import ensure_schema


async def main() -> None:
    client = Neo4jClient(get_settings())
    try:
        if not await client.health():
            raise SystemExit("Neo4j is unavailable; start it with `docker compose up -d neo4j`")
        await ensure_schema(client)
        print("ScopeGraph schema is ready.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

