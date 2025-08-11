import asyncio
import dataclasses
import logging
import time
import typing
from dataclasses import field
from datetime import datetime
from itertools import zip_longest
from typing import Optional

import discord
import httpx
from discord.app_commands import Choice
from discord.ext import commands
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ImageUrl, AudioUrl, VideoUrl, DocumentUrl, \
    ModelResponse, UserPromptPart, TextPart, UserContent, PartDeltaEvent, PartStartEvent, BinaryContent, \
    ModelRequestPart, ModelResponsePart, ToolCallPart
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP

import gemini_live
from config import get_config, config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

curr_model = next(iter(config["models"]))
max_text = config.get("max_text", 100000)
max_messages = config.get("max_messages", 25)

msg_nodes: 'dict[Any, MsgNode]' = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
activity = discord.CustomActivity(name=(config["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

def parse_mcp_option(name: str, option: dict):
    if "url" in option:
        return MCPServerStreamableHTTP(option["url"], tool_prefix=name, max_retries=option.get("max_retries", 5))
    else:
        option.setdefault("tool_prefix", name)
        return MCPServerStdio(**option)

toolsets: list[AbstractToolset] = [
    parse_mcp_option(name, mcp)
    for name, mcp in config.get("mcpServers", {}).items()
]

httpx_client = httpx.AsyncClient()


@dataclasses.dataclass
class MsgNode:
    msg: Optional[list[ModelMessage]] = None

    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def discord_attachment_to_fileurl(att: discord.Attachment) -> BinaryContent | ImageUrl | AudioUrl | VideoUrl | DocumentUrl:
    if att.content_type.startswith("image"):
        # Force fetch
        return BinaryContent(data=(await httpx_client.get(att.url)).content, media_type=att.content_type)
        # return ImageUrl(att.url, media_type=att.content_type)
    elif att.content_type.startswith("audio"):
        return AudioUrl(att.url, media_type=att.content_type)
    # Not supported by OpenAI
    # elif att.content_type.startswith("video"):
    #     return VideoUrl(att.url, media_type=att.content_type)
    else:
        return DocumentUrl(att.url, media_type=att.content_type)


async def discord_msg_to_modelmessage(msg: discord.Message, max_images: int) -> ModelMessage:
    if msg.author == discord_bot.user:
        out = ModelResponse(parts=[])
    else:
        out = ModelRequest(parts=[])

    cleaned_content = msg.content.removeprefix(discord_bot.user.mention).lstrip()
    text = "\n".join(
        ([cleaned_content] if cleaned_content else [])
        + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in msg.embeds]
    )

    if len(text) > max_text:
        text = text[:max_text]
        # TODO: Warnings
        # user_warnings.add(f"⚠️ Max {max_text:,} characters per message")

    if msg.author != discord_bot.user:
        content: list[UserContent] = [text]

        attachments = await asyncio.gather(*[discord_attachment_to_fileurl(att) for att in msg.attachments])
        for att in attachments:
            if not isinstance(att, DocumentUrl):
                content.append(att)
            else:
                # user_warnings.add("⚠️ Unsupported attachments")
                pass

        if len(content) > max_images + 1:
            # user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            content = content[:max_images+1]

        # curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
        out.parts.append(UserPromptPart(content=content))
    else:
        out.parts.append(TextPart(text))

    return out

@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    choices = [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()][:24]
    choices += [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []

    return choices


@discord_bot.event
async def on_ready() -> None:
    if client_id := config["client_id"]:
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

    await discord_bot.tree.sync()

def get_agent(new_msg: discord.Message) -> Agent:
    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")

    extra_headers = provider_config.get("extra_headers", None) or {}
    extra_body = provider_config.get("extra_body", None) or {}

    model_parameters = config["models"].get(provider_slash_model, None) or {}
    model_parameters["extra_headers"] = extra_headers
    model_parameters["extra_body"] = extra_body

    agent_kwargs = {}

    provider = OpenAIProvider(base_url=base_url, api_key=api_key, http_client=httpx_client)
    model = OpenAIModel(model_name=model, provider=provider, settings=ModelSettings(**model_parameters))

    system_prompt = None
    if "system_prompt" in config:
        now = datetime.now().astimezone()

        accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)
        system_prompt = [
            config["system_prompt"]
                .replace("{id}", discord_bot.user.mention)
                .replace("{user_id}", new_msg.author.mention)
                .replace("{date}", now.strftime("%B %d %Y"))
                .replace("{time}", now.strftime("%H:%M:%S %Z%z"))
                .strip()
         ]
        if accept_usernames:
            system_prompt.append("User's names are their Discord IDs and should be typed as '<@ID>'")

    support_tool_use = model_parameters.get("tools", False)
    if support_tool_use:
        def get_user():
            """Get the user information of the last message"""
            return f"Last message's author name: {new_msg.author.name}\nWhen mentioning this user's full name, ALWAYS use the mention tag {new_msg.author.mention} (with <>) instead of their name\nSubsequent messages may have different names"
        agent_kwargs['tools'] = [get_user]
        agent_kwargs['toolsets'] = toolsets

    agent = Agent(
        model=model,
        instructions=system_prompt,
        output_type=str,
        retries=model_parameters.get("retries", 5),
        **agent_kwargs,
    )
    agent.image_support = model_parameters.get("image", False) or any(x in agent.model.model_name for x in VISION_MODEL_TAGS)

    return agent

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    config = await asyncio.to_thread(get_config)

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    agent = get_agent(new_msg)

    accept_images = typing.cast(typing.Any, agent).image_support
    max_images = config.get("max_images", 5) if accept_images else 0

    # Build message chain and set user warnings
    messages: list[ModelMessage] = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.msg is None:
                curr_node.msg = [await discord_msg_to_modelmessage(curr_msg, max_images)]
                # TODO: Warnings
                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            messages.extend(curr_node.msg)
            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    edit_task = None
    response_msgs: list[discord.Message] = []

    async def update_reply(message: str, incomplete=False, force_flush=False):
        """
        Create reply to user's message, or update existing reply (within rate limit)
        Also manage the split of messages when cap is hit
        """
        nonlocal response_msgs, edit_task
        global last_task_time

        if incomplete:
            message += STREAMING_INDICATOR

        message_left = message
        message_parts = []
        while len(message_left) > 0:
            message_parts.append(message_left[:max_message_length])
            message_left = message_left[max_message_length:]

        for (index, (part_message, discord_msg)) in enumerate(zip_longest(message_parts, response_msgs)):
            embed = discord.Embed()
            embed.description = part_message
            if index == 0:
                for warning in sorted(user_warnings):
                    embed.add_field(name=warning, value="", inline=False)

            embed.color = EMBED_COLOR_COMPLETE
            if incomplete:
                embed.color = EMBED_COLOR_INCOMPLETE

            # (Some, None) = Create
            # (Some, Some) = Update
            # (None, Some) = Delete
            # (None, None) = Should not happen
            if part_message is not None and discord_msg is None: # Create
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                discord_msg = await reply_to_msg.reply(embed=embed, silent=True)
                response_msgs.append(discord_msg)

                msg_nodes[discord_msg.id] = MsgNode(parent_msg=new_msg)
                await msg_nodes[discord_msg.id].lock.acquire()
            elif part_message is not None and discord_msg is not None: # Update
                if discord_msg.content != part_message:
                    ready_to_edit = (edit_task is None or edit_task.done()) and time.monotonic() - last_task_time >= EDIT_DELAY_SECONDS
                    if ready_to_edit or not incomplete or force_flush:
                        if edit_task is not None:
                            await edit_task

                        discord_msg.content = part_message
                        edit_task = asyncio.create_task(discord_msg.edit(embed=embed))
                        last_task_time = time.monotonic()
            elif part_message is None and discord_msg is not None: # Delete
                await discord_msg.delete()
                response_msgs = [item for item in response_msgs if item.id != discord_msg.id]

    try:
        async with new_msg.channel.typing():
            async with agent:
                async with agent.iter(messages[0].parts[0].content, message_history=messages[1:][::-1]) as run:
                    agent_messages = []
                    async for node in run:
                        if Agent.is_model_request_node(node):
                            current_part = None
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    if isinstance(event, PartStartEvent):
                                        if current_part is not None:
                                            agent_messages.append([current_part])

                                        current_part = event.part
                                        # We don't commit now until the next
                                    elif isinstance(event, PartDeltaEvent):
                                        current_part = event.delta.apply(current_part)

                                    await update_reply(format_message_history(agent_messages + [[current_part]]), incomplete=True)

                            if current_part is not None:
                                agent_messages.append([current_part])

                    await update_reply(format_message_history([v.parts for v in run.result.new_messages()]))
                    new_messages = run.result.new_messages()[::-1]

        for response_msg in response_msgs:
            msg_nodes[response_msg.id].msg = new_messages
            msg_nodes[response_msg.id].lock.release()
    except Exception:
        logging.exception("Error while generating response")
        await update_reply("An error occurred while generating response")

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)

def format_message_history(parts: list[list[ModelRequestPart | ModelResponsePart]]) -> str:
    out = []

    for part in parts:
        for msg in part:
            if isinstance(msg, TextPart) and len(msg.content) > 0:
                out.append(msg.content)
            elif isinstance(msg, ToolCallPart):
                out.append(f"-# Using tool `{msg.tool_name}`")
            # We don't show thinking...

    return "\n\n".join(out)

async def main() -> None:
    if "voice" in config and config["voice"]["enabled"]:
        discord_bot.tree.add_command(gemini_live.live_command)
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
