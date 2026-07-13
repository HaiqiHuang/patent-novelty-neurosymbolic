import os

import epo_ops
from dotenv import load_dotenv


def get_epo_client() -> epo_ops.Client:
    """
    Create an EPO OPS client using credentials stored in .env.

    Required .env variables:
        EPO_OPS_KEY
        EPO_OPS_SECRET
    """
    load_dotenv()

    key = os.getenv("EPO_OPS_KEY")
    secret = os.getenv("EPO_OPS_SECRET")

    if not key or not secret:
        raise RuntimeError(
            "Missing EPO_OPS_KEY or EPO_OPS_SECRET. "
            "Please add them to your .env file."
        )

    return epo_ops.Client(key=key, secret=secret)