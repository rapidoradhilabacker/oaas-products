from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
import jwt
from app.schemas import TokenRequest, TokenResponse, GenericResponse, ErrorCode
from app.config import AUTH_SETTINGS
from http import HTTPStatus
from app.tracing import tracer

auth_router = APIRouter()

@auth_router.post("/token", response_model=TokenResponse)
async def generate_token(request: TokenRequest):
    """
    Generate an authentication token with 3 months validity.
    
    Args:
        request: TokenRequest containing the service_id
        
    Returns:
        TokenResponse: Contains the generated JWT token and expiration date
        
    Raises:
        HTTPException: If the service_id is invalid or token generation fails
    """
    with tracer.start_as_current_span("generate_token") as span:
        # Validate the service_id
        if request.service_id != AUTH_SETTINGS.service_id:
            error_response = GenericResponse.get_error_response(
                error_code=ErrorCode.ERROR_CODE_AUTH_ERROR,
                customer_message='Invalid service_id',
                debug_info={}
            )
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail=error_response.dict(),
            )
        
        try:
            # Set token expiration to 3 months from now
            current_time = datetime.utcnow()
            expires_at = current_time + timedelta(days=90)
            
            # Create the JWT payload
            payload = {
                "sub": request.service_id,
                "iat": current_time,
                "exp": expires_at
            }
            
            # Generate the token
            token = jwt.encode(
                payload, 
                AUTH_SETTINGS.secret_key, 
                algorithm=AUTH_SETTINGS.algorithm
            )
            
            # Format the expiration date as ISO string
            expires_at_str = expires_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            return TokenResponse(token=token, expires_at=expires_at_str)
        
        except Exception as e:
            span.record_exception(e)
            error_response = GenericResponse.get_error_response(
                error_code=ErrorCode.ERROR_CODE_UNKNOWN,
                customer_message='Failed to generate token',
                debug_info={"error": str(e)}
            )
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=error_response.dict(),
            )
