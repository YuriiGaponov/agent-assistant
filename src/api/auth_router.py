"""src.api.auth_router"""

from fastapi import APIRouter

from src.users import fastapi_users, auth_backend

router = APIRouter(
    tags=['auth']
)

router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
)
