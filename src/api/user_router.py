"""src.api.user_router"""

from fastapi import APIRouter

from src.users import fastapi_users, UserRead, UserCreate

router = APIRouter(
    tags=['user']
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
)
