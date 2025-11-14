from pydantic import BaseModel, Field
from typing import Optional


class SSHConfig(BaseModel):
    hostname: str = Field(..., description="Remote server IP or domain")
    username: str = Field(..., description="SSH username")
    password: Optional[str] = Field(
        None, description="SSH password (optional)"
    )
    key_filename: Optional[str] = Field(
        None, description="Path to the private key file"
    )
    port: int = Field(22, description="SSH port")