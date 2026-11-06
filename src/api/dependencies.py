"""FastAPI dependencies: who is calling, and the shared chat service."""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from src.services.chat_service import ChatService

DEFAULT_DEV_USER = "dev-user"
MAX_USER_ID_LENGTH = 100


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_user_id(x_user_id: Annotated[str | None, Header()] = None) -> str:
    """TEMPORARY: reads the user from an X-User-Id header.

    Real Azure AD authentication replaces this later (Security step). The user id must then
    come from the verified token - never from something the browser can set freely.
    """
    user_id = (x_user_id or DEFAULT_DEV_USER).strip()
    if not user_id or len(user_id) > MAX_USER_ID_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid X-User-Id header.")
    return user_id


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
UserIdDep = Annotated[str, Depends(get_user_id)]
