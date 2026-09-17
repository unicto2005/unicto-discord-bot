import logging

from discord.ext import commands


logger = logging.getLogger("nova")


class NOVAEvents(commands.Cog):
    """NOVA Discord event handlers."""

    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(
            "NOVA is online as %s",
            self.bot.user,
        )

        logger.info(
            "Bot ID: %s",
            self.bot.user.id,
        )

    @commands.Cog.listener()
    async def on_message(self, message):
        logger.info(
            "📩 MESSAGE EVENT RECEIVED | author=%s | guild=%s | content=%r",
            message.author,
            message.guild,
            message.content,
        )

        # Ignore messages sent by bots.
        if message.author.bot:
            return
    
        # Commands are handled by discord.py's default command processor.
        if message.content.startswith("!"):
            return

        # Handle DMs.
        if message.guild is None:
            handler = getattr(
                self.bot,
                "nova_ai_handler",
                None,
            )

            if handler:
                await handler(
                    message,
                    message.content,
                )

            return

        # Handle messages that mention NOVA.
        if self.bot.user and self.bot.user.mentioned_in(message):
            prompt = message.content

            prompt = prompt.replace(
                f"<@{self.bot.user.id}>",
                "",
            ).replace(
                f"<@!{self.bot.user.id}>",
                "",
            ).strip()

            if not prompt:
                await message.reply(
                    "👋 Hey! I'm NOVA. What can I help you with?",
                    mention_author=False,
                )
                return

            handler = getattr(
                self.bot,
                "nova_ai_handler",
                None,
            )

            if handler:
                await handler(
                    message,
                    prompt,
                )


async def setup(bot):
    await bot.add_cog(NOVAEvents(bot))
