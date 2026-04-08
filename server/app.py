# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import PetclinicAction, PetclinicObservation
    from .petclinic_env_environment import PetclinicEnvironment
except (ImportError, ModuleNotFoundError):
    from models import PetclinicAction, PetclinicObservation
    from server.petclinic_env_environment import PetclinicEnvironment

# Create the app
app = create_app(
    PetclinicEnvironment,
    PetclinicAction,
    PetclinicObservation,
    env_name="petclinic_env",
    max_concurrent_envs=1,
)

def main():
    """
    Standardized entry point for the validator.
    """
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()