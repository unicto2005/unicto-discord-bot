import asyncio
import logging
import os

import discord
from dotenv import load_dotenv

from app.ai.router import GeminiRouter
from app.bot.client import NOVAClient
from app.health import start_health_server
from app.services.context_manager import context_manager
from app.services.memory import memory
from app.services.request_queue import request_queue
from app.services.team import team_service
from app.services.usage_guard import usage_guard


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("nova")


# ============================================================
# GEMINI
# ============================================================

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

gemini_router = GeminiRouter(
    api_key=GEMINI_API_KEY,
    models=MODELS,
)


async def ask_nova(prompt: str, conversation_context: str = ""):
    """
    Send a request to Gemini through the NOVA model router.
    """

    return await gemini_router.ask(
        prompt=prompt,
        conversation_context=conversation_context,
    )


# ============================================================
# DISCORD BOT
# ============================================================

bot = NOVAClient(request_queue)


# ============================================================
# DISCORD HELPERS
# ============================================================

def get_author(source):
    """Safely get the Discord user/member from a message or interaction."""

    if isinstance(source, discord.Interaction):
        return source.user

    return source.author


def get_channel(source):
    """Safely get the Discord channel from a message or interaction."""

    if isinstance(source, discord.Interaction):
        return source.channel

    return source.channel


async def send_reply(source, content: str):
    """
    Send a Discord reply while respecting Discord's 2000-character limit.
    """

    if not content:
        content = "I couldn't generate a response."

    chunks = [
        content[i:i + 2000]
        for i in range(0, len(content), 2000)
    ]

    if isinstance(source, discord.Interaction):

        if not source.response.is_done():
            await source.response.send_message(chunks[0])

            for chunk in chunks[1:]:
                await source.followup.send(chunk)

        else:
            for chunk in chunks:
                await source.followup.send(chunk)

    else:

        for chunk in chunks:
            await source.reply(chunk)


# ============================================================
# AI REQUEST PROCESSING
# ============================================================

async def process_ai_request(
    source,
    prompt: str,
):
    """
    Submit an AI request to NOVA's request queue.
    """

    return await request_queue.submit(
        process_ai_request_inner,
        source,
        prompt,
    )


async def process_ai_request_inner(
    source,
    prompt: str,
):
    """
    Main AI request pipeline.
    """

    author = get_author(source)
    channel = get_channel(source)

    user_id = author.id
    channel_id = str(channel.id) if channel else None

    username = getattr(author, "display_name", None)

    if not username:
        username = getattr(author, "name", "Unknown User")

    # --------------------------------------------------------
    # Usage guard
    # --------------------------------------------------------

    allowed = await usage_guard.check(user_id)

    if not allowed:
        await send_reply(
            source,
            "You've reached your current NOVA usage limit. Please try again later.",
        )
        return

    # --------------------------------------------------------
    # Register team member
    # --------------------------------------------------------

    try:
        await team_service.register_member(
            user_id=user_id,
            username=username,
        )
    except Exception:
        logger.exception("Failed to register team member.")

    # --------------------------------------------------------
    # Load memory
    # --------------------------------------------------------

    try:
        user_memory = await memory.get_user_memory(user_id)
    except Exception:
        logger.exception("Failed to load user memory.")
        user_memory = ""

    # --------------------------------------------------------
    # Build conversation context
    # --------------------------------------------------------

    try:
        conversation_context = await context_manager.build_context(
            user_id=user_id,
            channel_id=channel_id,
            prompt=prompt,
            memory=user_memory,
        )
    except Exception:
        logger.exception("Failed to build conversation context.")
        conversation_context = ""

    # --------------------------------------------------------
    # Ask Gemini
    # --------------------------------------------------------

    try:
        response = await ask_nova(
            prompt=prompt,
            conversation_context=conversation_context,
        )

    except Exception:
        logger.exception("Gemini request failed.")

        await send_reply(
            source,
            "I'm having trouble reaching my AI system right now. Please try again in a moment.",
        )
        return

    # --------------------------------------------------------
    # Normalize response
    # --------------------------------------------------------

    if response is None:
        response = "I couldn't generate a response right now."

    if not isinstance(response, str):
        response = str(response)

    # --------------------------------------------------------
    # Record usage
    # --------------------------------------------------------

    try:
        await usage_guard.record(user_id)
    except Exception:
        logger.exception("Failed to record usage.")

    # --------------------------------------------------------
    # Save conversation
    # --------------------------------------------------------

    try:
        await memory.save_message(
            user_id=user_id,
            role="user",
            content=prompt,
        )

        await memory.save_message(
            user_id=user_id,
            role="assistant",
            content=response,
        )

    except Exception:
        logger.exception("Failed to save conversation memory.")

    # --------------------------------------------------------
    # Send response
    # --------------------------------------------------------

    await send_reply(
        source,
        response,
    )


# ============================================================
# CONNECT AI HANDLER TO BOT
# ============================================================

bot.nova_ai_handler = process_ai_request


# ============================================================
# DISCORD CONNECTION
# ============================================================

async def run_nova():
    """
    Start NOVA and maintain its Discord connection.

    Discord HTTP 429 errors are logged with endpoint and
    rate-limit information so we can diagnose Cloudflare/
    Discord blocking without repeatedly hammering the API.
    """

    await start_health_server()

    logger.info("Health server started successfully.")

    rate_limit_attempts = 0

    while True:

        try:
            logger.info("Connecting NOVA to Discord...")

            await bot.start(DISCORD_TOKEN)

            logger.warning(
                "NOVA Discord connection closed."
            )

            # A clean connection means we can reset the
            # rate-limit backoff counter.
            rate_limit_attempts = 0

        except discord.HTTPException as error:

            status = getattr(error, "status", None)
            retry_after = getattr(error, "retry_after", None)
            response = getattr(error, "response", None)

            response_url = getattr(response, "url", None)
            response_headers = getattr(response, "headers", {}) or {}

            content_type = response_headers.get(
                "Content-Type"
            )

            ratelimit_limit = response_headers.get(
                "X-RateLimit-Limit"
            )

            ratelimit_remaining = response_headers.get(
                "X-RateLimit-Remaining"
            )

            ratelimit_reset = response_headers.get(
                "X-RateLimit-Reset-After"
            )

            header_retry_after = response_headers.get(
                "Retry-After"
            )

            error_text = getattr(
                error,
                "text",
                "",
            )

            if error_text:
                error_text = str(error_text)[:500]

            logger.error(
                "Discord HTTP error | "
                "type=%s | "
                "status=%s | "
                "retry_after=%s | "
                "url=%s | "
                "content_type=%s | "
                "rate_limit=%s | "
                "remaining=%s | "
                "reset_after=%s | "
                "header_retry_after=%s | "
                "body=%s",
                type(error).__name__,
                status,
                retry_after,
                response_url,
                content_type,
                ratelimit_limit,
                ratelimit_remaining,
                ratelimit_reset,
                header_retry_after,
                error_text,
            )

            # ------------------------------------------------
            # Rate limited
            # ------------------------------------------------

            if status == 429:

                rate_limit_attempts += 1

                # Discord normally supplies retry_after.
                # If it doesn't, use an increasing backoff
                # so Render doesn't repeatedly hit Discord.
                base_wait = (
                    retry_after
                    if retry_after is not None
                    else 60
                )

                backoff = min(
                    60 * (2 ** (rate_limit_attempts - 1)),
                    600,
                )

                wait_time = max(
                    float(base_wait),
                    float(backoff),
                )

                logger.warning(
                    "Discord rate limited NOVA. "
                    "Attempt #%d. "
                    "Waiting %.1f seconds before retrying...",
                    rate_limit_attempts,
                    wait_time,
                )

                await asyncio.sleep(wait_time)

                continue

            # ------------------------------------------------
            # Other Discord HTTP errors
            # ------------------------------------------------

            logger.error(
                "Non-retryable Discord HTTP error."
            )

            raise

        except discord.LoginFailure:

            logger.exception(
                "NOVA failed Discord authentication. "
                "Check the DISCORD_TOKEN."
            )

            raise

        except discord.GatewayNotFound:

            logger.exception(
                "Discord Gateway could not be reached. "
                "Retrying in 30 seconds..."
            )

            await asyncio.sleep(30)

        except discord.ConnectionClosed as error:

            logger.warning(
                "Discord Gateway connection closed | "
                "code=%s | reason=%s",
                getattr(error, "code", None),
                error,
            )

            logger.info(
                "Retrying Discord connection in 30 seconds..."
            )

            await asyncio.sleep(30)

        except asyncio.CancelledError:

            logger.info(
                "NOVA shutdown requested."
            )

            raise

        except Exception:

            logger.exception(
                "Unexpected NOVA error. "
                "Retrying in 30 seconds..."
            )

            await asyncio.sleep(30)


# ============================================================
# MAIN
# ============================================================

def main():
    """Application entry point."""

    logger.info("========================================")
    logger.info("Starting NOVA...")
    logger.info("========================================")

    try:
        asyncio.run(run_nova())

    except KeyboardInterrupt:

        logger.info(
            "NOVA stopped by user."
        )


if __name__ == "__main__":
    main()
