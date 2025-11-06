import subprocess
import sys
import importlib
import os
import time
import json
import re
import asyncio
from datetime import datetime, timedelta, timezone  # ← Added timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import hashlib
import base64

# ============================================================================
# AUTO-INSTALL SYSTEM - Install external packages ONLY
# ============================================================================
def install_and_import(package_name, import_name=None):
    """Auto-install external packages (NOT core uagents packages)"""
    if import_name is None:
        import_name = package_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return importlib.import_module(import_name)
        except Exception as e:
            print(f"⚠️ Failed to install {package_name}: {e}")
            return None

# Install external dependencies (ONLY if needed)
requests = install_and_import("requests")

# ============================================================================
# PROTOCOL IMPORTS - NO FALLBACKS, NO TRY/EXCEPT
# ============================================================================
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    EndSessionContent,
    chat_protocol_spec
)
from uagents import Agent, Context, Protocol
from uagents.storage import StorageAPI

# ============================================================================
# PRODUCTION-READY CLASSES
# ============================================================================

class RateLimiter:
    """Rate limiting with sliding window"""
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: List[float] = []

    async def acquire(self) -> bool:
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.window]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

    def remaining(self) -> int:
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.window]
        return max(0, self.max_requests - len(self.requests))


class LRUCache:
    """LRU cache with TTL"""
    def __init__(self, max_size: int = 100, ttl_seconds: int = 900):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            self.access_order.remove(key)
            return None
        self.access_order.remove(key)
        self.access_order.append(key)
        return value

    def set(self, key: str, value: Any):
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = (value, time.time())
        self.access_order.append(key)


class ASI1MiniClient:
    """Production ASI1-mini client with rate limiting and caching"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "sk_2a3c92a0b11e4f18b50708cca1a55179ab38a7c2fb7f4eee95fd68e1e28f860b"
        self.base_url = "https://api.asi1.ai/v1"
        self.model = "asi1-mini"
        self.rate_limiter = RateLimiter(60, 60)  # 60 requests per minute
        self.cache = LRUCache(100, 900)  # 15-minute cache, 100 items
        self.circuit_breaker = {"failures": 0, "last_failure": 0, "threshold": 5, "reset_timeout": 300}
        self.enabled = requests is not None

    def _check_circuit_breaker(self) -> bool:
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            if time.time() - self.circuit_breaker["last_failure"] < self.circuit_breaker["reset_timeout"]:
                return False  # Circuit is open
            self.circuit_breaker["failures"] = 0  # Reset after timeout
        return True

    async def _make_api_call(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        if not self.enabled or not requests:
            return None

        if not self._check_circuit_breaker():
            return None

        if not await self.rate_limiter.acquire():
            return None

        cache_key = hashlib.md5(f"{prompt}:{max_tokens}".encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()['choices'][0]['message']['content']
                self.cache.set(cache_key, result)
                self.circuit_breaker["failures"] = max(0, self.circuit_breaker["failures"] - 1)
                return result
            elif response.status_code == 429:
                await asyncio.sleep(1)
                return None
            else:
                self.circuit_breaker["failures"] += 1
                self.circuit_breaker["last_failure"] = time.time()
                return None
        except Exception as e:
            self.circuit_breaker["failures"] += 1
            self.circuit_breaker["last_failure"] = time.time()
            return None

    async def process(self, query: str) -> str:
        if self.enabled:
            response = await self._make_api_call(query, max_tokens=500)
            if response:
                return response
        return "I understand your query but AI processing is currently unavailable."


class PersistentStorageManager:
    """Production storage with atomic operations and rollback"""
    def __init__(self, storage: StorageAPI):
        self.storage = storage
        self._init_storage()

    def _init_storage(self):
        """Initialize storage structure"""
        if self.storage.get("initialized") is None:
            self.storage.set("initialized", True)
            self.storage.set("version", "1.0")
            self.storage.set("created_at", datetime.now(timezone.utc).isoformat())
            self.storage.set("main_data", {})
            self.storage.set("cache", {})
            self.storage.set("metadata", {"total_requests": 0, "errors": 0})

    def get(self, key: str, default=None) -> Any:
        """Get value from storage — handle default manually"""
        value = self.storage.get(key)
        return value if value is not None else default

    def set(self, key: str, value: Any) -> bool:
        """Set value in storage with backup"""
        try:
            backup_key = f"_backup_{key}"
            old_value = self.storage.get(key)
            if old_value is not None:
                self.storage.set(backup_key, old_value)
            self.storage.set(key, value)
            return True
        except Exception:
            return False

    def rollback(self, key: str) -> bool:
        """Rollback to previous value"""
        try:
            backup_key = f"_backup_{key}"
            backup = self.storage.get(backup_key)
            if backup is not None:
                self.storage.set(key, backup)
                return True
            return False
        except Exception:
            return False

    def increment_counter(self, counter_name: str) -> int:
        """Atomically increment a counter"""
        metadata = self.get("metadata", {})
        metadata[counter_name] = metadata.get(counter_name, 0) + 1
        self.set("metadata", metadata)
        return metadata[counter_name]


# ============================================================================
# AGENT INITIALIZATION - CRITICAL: NO PARAMETERS!
# ============================================================================
agent = Agent()  # ← MUST BE EXACTLY THIS, NO PARAMETERS!

# Initialize storage manager using agent's built-in storage
storage_mgr = PersistentStorageManager(agent.storage)

# Initialize AI client
ai_client = ASI1MiniClient()

# ============================================================================
# CHAT PROTOCOL - MUST USE spec PARAMETER
# ============================================================================
chat_proto = Protocol(spec=chat_protocol_spec)

def create_text_chat(text: str) -> ChatMessage:
    """Helper to create text chat messages"""
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)]
    )

@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages with MANDATORY acknowledgement"""

    # STEP 1: IMMEDIATE ACKNOWLEDGEMENT (MANDATORY - MUST BE FIRST!)
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc),
        acknowledged_msg_id=msg.msg_id
    ))

    # STEP 2: Extract text from message
    query = ""
    for content in msg.content:
        if hasattr(content, 'text'):
            query += content.text + " "

    # STEP 3: Clean metadata
    query = re.sub(r'\[additional context\].*$', '', query, flags=re.IGNORECASE | re.DOTALL)
    query = re.sub(r'<user_details>.*?</user_details>', '', query, flags=re.DOTALL)
    query = re.sub(r'<knowledge_graph>.*?</knowledge_graph>', '', query, flags=re.DOTALL)
    query = query.strip()

    # STEP 4: Log request
    ctx.logger.info(f"Processing query from {sender}: {query[:100]}")
    storage_mgr.increment_counter("total_requests")

    # STEP 5: Process with AI
    response_text = ""
    if "send email" in query.lower():
        response_text = (
            "Okay, I understand you'd like to send an email. For now, I'm an interface. "
            "You'll need to integrate me with your PostMouse JavaScript agent's API to trigger email sending."
        )
    elif "status" in query.lower():
        response_text = (
            "PostMouseController is running. To check the status of your PostMouse JavaScript agent, "
            "you'd need to implement an API call to it here."
        )
    else:
        response_text = await ai_client.process(
            f"Provide a brief, helpful response about an email agent named "
            f"PostMouseController. User query: {query}"
        )

    if not response_text:
        response_text = (
            "I am PostMouseController, an Agentverse interface for your PostMouse agent. "
            "How can I help you manage your email tasks?"
        )

    # STEP 6: Send response with EndSessionContent (REQUIRED)
    await ctx.send(sender, ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=str(uuid4()),
        content=[
            TextContent(type="text", text=response_text),
            EndSessionContent(type="end-session")
        ]
    ))


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgements"""
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")


# Include protocol with manifest publishing
agent.include(chat_proto, publish_manifest=True)

# ============================================================================
# LIFECYCLE HANDLERS
# ============================================================================

@agent.on_event("startup")
async def on_startup(ctx: Context):
    """Agent startup handler"""
    ctx.logger.info("🚀 Agent started!")
    ctx.logger.info(f"📬 Address: {agent.address}")
    ctx.logger.info("🔧 Storage initialized")

    if ai_client.enabled and ai_client._check_circuit_breaker():
        ctx.logger.info("✅ ASI1 AI client ready")
    else:
        ctx.logger.warning("⚠️ ASI1 AI client unavailable - using fallback responses")


@agent.on_interval(period=300.0)  # Every 5 minutes
async def periodic_update(ctx: Context):
    """Periodic maintenance tasks"""
    metadata = storage_mgr.get("metadata", {})
    total_requests = metadata.get("total_requests", 0)
    errors = metadata.get("errors", 0)
    ctx.logger.info(f"📊 Stats: {total_requests} requests, {errors} errors")


@agent.on_event("shutdown")
async def on_shutdown(ctx: Context):
    """Agent shutdown handler"""
    ctx.logger.info("👋 Agent shutting down...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    agent.run()