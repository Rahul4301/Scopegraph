from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncTransaction

from scopegraph.config import Settings


class Neo4jClient:
    """Small async driver wrapper; Cypher remains in the graph layer."""

    def __init__(self, settings: Settings) -> None:
        self._database = settings.neo4j_database
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )
        self._transaction: ContextVar[AsyncTransaction | None] = ContextVar(
            f"scopegraph-transaction-{id(self)}", default=None
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        if self._transaction.get() is not None:
            yield
            return
        async with self._driver.session(database=self._database) as session:
            transaction = await session.begin_transaction()
            token = self._transaction.set(transaction)
            try:
                yield
                await transaction.commit()
            except BaseException:
                await transaction.rollback()
                raise
            finally:
                self._transaction.reset(token)

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
        transaction = self._transaction.get()
        if transaction is not None:
            result = await transaction.run(query, dict(parameters or {}))
            return [record.data() async for record in result]
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, dict(parameters or {}))
            return [record.data() async for record in result]

    async def execute_read(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        transaction = self._transaction.get()
        if transaction is not None:
            result = await transaction.run(query, dict(parameters or {}))
            return [record.data() async for record in result]
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, dict(parameters or {}))
            return [record.data() async for record in result]

    async def run_statements(self, statements: Sequence[str]) -> None:
        for statement in statements:
            await self.execute_write(statement)
