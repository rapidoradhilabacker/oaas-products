from fastapi import Header, Request, HTTPException
from http import HTTPStatus
from app.config import AUTH_SETTINGS
from app.schemas import GenericResponse, Trace, ErrorCode
from app.tracing import tracer
import jwt
from fastapi import Depends

async def get_trace(
    x_request_id: str = Header(),
    x_device_id: str = Header()
) -> Trace:
    return Trace(
        request_id=x_request_id,
        device_id=x_device_id,

    )

async def get_current_user(
    request: Request, 
    x_request_id: str | None = Header(default=None), 
    x_device_id: str | None = Header(default=None),
    authorization: str = Header(),
    trace: Trace = Depends(get_trace)
):
    # body = await request.body()
    # body_str = body.decode('utf-8')
    headers = request.headers
    attributes: dict[str, str] = {
        # 'body': body_str,
        'token': str(authorization), 
        'request_id': headers.get('x-request-id', ''), 
        'device_id': x_device_id or '',
    }

    with tracer.start_as_current_span("get_request_param", attributes=attributes) as span:
        error_response = GenericResponse.get_error_response(
            error_code=ErrorCode.ERROR_CODE_AUTH_ERROR,
            customer_message='Invalid Token',
            debug_info={}
        )
        credentials_exception = HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail=error_response.dict(),  # Convert to dictionary for JSON serialization
        )
        if not authorization:
            raise credentials_exception
        try:
            token = authorization.split("Bearer ")[1]
            payload = jwt.decode(token, AUTH_SETTINGS.secret_key, algorithms=[AUTH_SETTINGS.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            elif username != AUTH_SETTINGS.service_id:
                raise credentials_exception
            
        except Exception:
            raise credentials_exception
        return trace