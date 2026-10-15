from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db

# Type-annotated database session dependency
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
