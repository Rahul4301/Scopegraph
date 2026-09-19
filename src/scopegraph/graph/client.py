from collections.abc import Mapping, Sequence
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from scopegraph.config import Settings


class Neo4jClient:
    """Small async driver wrapper; Cypher remains in the graph layer."""

    def __init__(self, settings: Settings) -> None:
        self._database = settings.neo4j_database
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )

    async def close(self) -> None:
        await self._driver.close()

    async def health(self) -> bool:
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    async def execute_write(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, dict(parameters or {}))
            return [record.data() async for record in result]

    async def execute_read(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, dict(parameters or {}))
            return [record.data() async for record in result]

    async def run_statements(self, statements: Sequence[str]) -> None:
        for statement in statements:
            await self.execute_write(statement)

