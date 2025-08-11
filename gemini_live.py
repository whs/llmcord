import asyncio
import io
import logging
from pickletools import read_uint1
from typing import Optional

import discord
from discord.ext import voice_recv, commands
from google import genai
from google.genai import types, live
from queuepipeio import PipeWriter, PipeReader

from config import config


@discord.app_commands.command(name="live", description="Make the bot join your voice channel")
async def live_command(interaction: discord.Interaction) -> None:
    if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You are not admin", ephemeral=True)
        return

    if not interaction.guild:
        await interaction.response.send_message("This command must be run in a server", ephemeral=True)
        return

    for vc in interaction.guild.voice_channels:
        if interaction.user in vc.members:
            asyncio.create_task(GeminiLiveConnection(interaction.client, config["voice"]).start(vc))
            await interaction.response.send_message(f"Joined {vc.mention}", ephemeral=True)
            return

    await interaction.response.send_message("Can't find your channel. Join a voice channel first", ephemeral=True)

class GeminiLiveConnection:
    voice_conn = None
    session: Optional[live.AsyncSession] = None

    def __init__(
        self,
        discord_bot: commands.Bot,
        config: dict,
        /,
        loop = None,
    ):
        self.bot = discord_bot
        self.config = config
        self.loop = loop or asyncio.get_running_loop()

        self.gemini_config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.config["voice"])),
            ),
            enable_affective_dialog=True,
            proactivity=types.ProactivityConfig(proactive_audio=True),
            system_instruction=self.config["system_prompt"],
            tools=[
            ]
        )

        self.gemini = genai.Client(vertexai=True)
        self.session = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bot.add_listener(self.on_voice_state_update, 'on_voice_state_update')

    async def start(self, channel: discord.VoiceChannel):
        self.channel = channel
        self.voice_conn = await self.channel.connect(cls=voice_recv.VoiceRecvClient)
        async with self.gemini.aio.live.connect(model=self.config["model"], config=self.gemini_config) as session:
            self.logger.info("Gemini connected")
            self.session = session

            sink = voice_recv.SilenceGeneratorSink(_GeminiFFmpegSink(session, asyncio.get_running_loop()))
            self.voice_conn.listen(sink)

            while self.voice_conn.is_listening():
                bot_output_w = PipeWriter(chunk_size=io.DEFAULT_BUFFER_SIZE)
                bot_output_r = PipeReader()
                bot_output_w.connect(bot_output_r)

                # play_future = asyncio.get_running_loop().create_future()
                # I can't get this to unbuffer (it seems that many place has io.DEFAULT_BUFFER_SIZE hardcoded)
                # so just recreate ffmpeg every time ensure that it will flush
                audio_source = discord.FFmpegOpusAudio(bot_output_r, pipe=True, before_options=f"-f s16le -ar 24000")
                # self.voice_conn.play(
                #     audio_source,
                #     # after=play_future.set_result,
                #     application="voip", signal_type="voice"
                # )

                async for response in session.receive():
                    if not response:
                        break

                    response_data = response.data
                    if response.server_content.interrupted is True:
                        self.logger.info("Turn interrupted")
                        # Flush audio source
                        self.voice_conn.stop_playing()
                        break
                    elif response_data is not None:
                        bot_output_w.write(response_data)

                        if self.voice_conn.source != audio_source:
                            self.voice_conn.stop_playing()
                            self.voice_conn.play(audio_source, application="voip", signal_type="voice")
                    elif response.go_away is not None:
                        await self.session.send_realtime_input(text=f"System: Time left in this session is {response.go_away.time_left}")
                    else:
                        self.logger.debug("Gemini response", response)

                bot_output_w.close()
                self.logger.info("Turn ended")

        await self.voice_conn.disconnect()

    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        if self.session is not None:
            if before.channel is None and after.channel == self.channel:
                asyncio.create_task(self.session.send_realtime_input(text=f"System: User \"{member.display_name}\" joined the voice channel"))

            if self.channel.members == [self.bot.user]:
                # Only me in here, leave
                await self.voice_conn.disconnect()
                await self.session.close()

class _GeminiLivePipe(io.BytesIO):
    def __init__(self, session: live.AsyncSession, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.session = session
        self.loop = loop
        self.logger = logging.getLogger(self.__class__.__name__)

    def write(self, b: bytes):
        if b:
            async def send():
                await self.session.send_realtime_input(audio=types.Blob(
                    data=b, mime_type="audio/pcm;rate=16000"
                ))

            asyncio.run_coroutine_threadsafe(send(), self.loop)

class _GeminiFFmpegSink(voice_recv.FFmpegSink):
    active_speaker = None

    def __init__(self, session: live.AsyncSession, loop: asyncio.AbstractEventLoop):
        super().__init__(buffer=_GeminiLivePipe(session, loop), options=f"-f s16le -ar 16000 -ac 1 -fflags flush_packets")

        self.session = session
        self.loop = loop
        self.logger = logging.getLogger(self.__class__.__name__)

    def write(self, user: Optional[discord.User], data: voice_recv.VoiceData):
        if self.active_speaker is None:
            self.active_speaker = user

        if self.active_speaker != user:
            return

        super().write(user, data)

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: discord.Member):
        if member == self.active_speaker:
            self.active_speaker = None
