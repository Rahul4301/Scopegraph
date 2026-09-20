"""Reset the local development graph after explicit confirmation."""

import argparse
import asyncio

from scopegraph.config import get_settings
from scopegraph.graph.client import Neo4jClient


async def _reset() -> None:
    client = Neo4jClient(get_settings())
    try:
        if not await client.health():
            raise SystemExit("Neo4j is unavailable; start it with `make neo4j-up`")
        await client.execute_write("MATCH (n) DETACH DELETE n")
        print("Development graph reset.")
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="confirm destructive reset")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to reset without --yes")
    asyncio.run(_reset())


if __name__ == "__main__":
    main()
