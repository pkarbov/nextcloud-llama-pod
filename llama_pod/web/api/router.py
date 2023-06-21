from fastapi.routing import APIRouter

from llama_pod.web.api import activate, echo, models, monitoring, redis, settings

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
api_router.include_router(redis.router, prefix="/redis", tags=["redis"])
api_router.include_router(models.router, prefix="/v1/models", tags=["models"])
api_router.include_router(settings.router, prefix="/v1/settings", tags=["settings"])
api_router.include_router(activate.router, prefix="/v1/activate", tags=["activate"])
