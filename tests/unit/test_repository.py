import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import Base
from app.repositories.base import BaseRepository


class SampleItem(Base):
    """Temporary test model."""

    __tablename__ = "test_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(50), nullable=False)
    description = Column(String(100), nullable=True)


@pytest.mark.asyncio
async def test_base_repository_crud(test_db_session: AsyncSession, test_engine):
    """Verify BaseRepository CRUD operations on an async session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(SampleItem.metadata.create_all)

    repo = BaseRepository(SampleItem, test_db_session)

    # Create
    item1 = await repo.create(title="Doc 1", description="Description 1")
    assert item1.id is not None
    assert item1.title == "Doc 1"

    item2 = await repo.create(title="Doc 2", description="Description 2")

    # Get by ID
    fetched = await repo.get_by_id(item1.id)
    assert fetched is not None
    assert fetched.title == "Doc 1"

    # Count
    total = await repo.count()
    assert total == 2

    # List all
    all_items = await repo.list_all(skip=0, limit=10)
    assert len(all_items) == 2

    # Update
    updated = await repo.update(fetched, title="Updated Doc 1")
    assert updated.title == "Updated Doc 1"

    # Delete
    deleted = await repo.delete_by_id(item1.id)
    assert deleted is True

    # Confirm deletion
    post_delete = await repo.get_by_id(item1.id)
    assert post_delete is None
    assert await repo.count() == 1
