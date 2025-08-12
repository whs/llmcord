import functools
import logging
import typing

import discord
from discord.ext import commands
import httpx
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart

from .spec import load_card, TavernCardV1, TavernCardV2
from llmcord import MsgNode


class CharacterCardCog(commands.Cog):
    http_client = httpx.AsyncClient()

    def __init__(self, bot: discord.ext.commands.Bot, cache: dict[typing.Any, "MsgNode"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = bot
        self.cache = cache

    @discord.app_commands.command(description="Use a character card")
    async def character(self, interaction: discord.Interaction, file: discord.Attachment) -> None:
        if not file.content_type == "image/png":
            await interaction.response.send_message("Character card file must be PNG", ephemeral=True)
            return

        logging.info(f"Loading model card...")
        content = (await self.http_client.get(file.url)).content
        character_card = load_card(content)
        if not character_card:
            await interaction.response.send_message("File is not a character card", ephemeral=True)
            return

        if isinstance(character_card.root, TavernCardV1):
            character_card = character_card.to_v2()
        else:
            character_card = character_card.root

        character_card = typing.cast(TavernCardV2, character_card)
        system_prompt = []

        templatize = functools.partial(character_card.data.templatize, username=interaction.user.mention)

        if character_card.data.system_prompt:
            system_prompt.append(templatize(character_card.data.system_prompt))
        if character_card.data.description:
            system_prompt.append(templatize(character_card.data.description))
        if character_card.data.personality:
            system_prompt.append(f"{character_card.data.name}'s personality: {templatize(character_card.data.personality)}")
        if character_card.data.scenario:
            system_prompt.append(templatize(character_card.data.scenario))
        # TODO: mes_example

        system_prompt = "\n".join(system_prompt)

        prompt = await interaction.response.send_message(templatize(character_card.data.first_mes))
        self.cache[prompt.message_id] = MsgNode(
            msg=[
                ModelRequest(
                    parts=[
                        SystemPromptPart(content=system_prompt),
                    ],
                    instructions=system_prompt,
                ),
                ModelResponse(
                    parts=[
                        TextPart(content=templatize(character_card.data.first_mes)),
                    ],
                )
            ],
            override_system_prompt=True,
        )
