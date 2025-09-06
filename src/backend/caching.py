import os
from dotenv import load_dotenv
from typing import Any, Dict, Optional

import orjson
import hashlib
from redis import asyncio as aioredis


load_dotenv(override=False)
REDIS_URL = os.getenv("REDIS_URL")
_client: aioredis.Redis | None = None


def _loads(obj: bytes) -> Any:
    return orjson.loads(obj)

def _dumps(obj: Any) -> bytes:
    return orjson.dumps(obj)

async def init() -> None:
    global _client

    _client = aioredis.from_url(REDIS_URL, decode_responses=False,
                                socket_connect_timeout=2, socket_timeout=2)
    await _client.ping()

async def close() -> None:
    global _client

    if _client is None:
        return

    try:
        await _client.close()
        _client = None
    except Exception as e:
        print(f"Could not close redis, {e}")

async def ping() -> bool:
    if _client is None:
        return False
    else:
        try:
            return bool(await _client.ping())
        except Exception as e:
            print(f"Can not reach redis client, {e}")
            return False

def _get_key(title: Optional[str], author: Optional[str], mood: Optional[str]) -> str:
    title = (title or "").strip().lower()
    author = (author or "").strip().lower()
    mood = (mood or "").strip().lower()
    blob = orjson.dumps([title, author, mood])
    h = hashlib.sha256(blob).hexdigest()[:32]
    return f"recs:{h}"

async def put(payload: Dict[str, Any]) -> None:
    title = payload.get("title", None)
    author = payload.get("author", None)
    mood = payload.get("mood", None)
    recs = payload.get("recs", None)

    if recs is None or _client is None:
        return

    key = _get_key(title, author, mood)
    await _client.set(key, _dumps(recs), ex=600)

async def get(title: Optional[str], author: Optional[str], mood: Optional[str]) -> Optional[Dict[str, str]]:
    if _client is None:
        return None

    key = _get_key(title, author, mood)
    b = await _client.get(key)
    return _loads(b) if b else None
