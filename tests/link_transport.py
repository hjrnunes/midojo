"""The test transport for the ``midojo-binding-v1`` link: an in-memory duplex, a fake clock and loss injection.

One :class:`LinkSession` is one binding: an anyio memory-stream pair carries
the frames, a :class:`LinkDispatcher` applies them to the control plane's
session on this side, and the :class:`~midojo_sdk.link.Link` plus its
recorder and world are the service side. Nothing here opens a socket.

A test injects a loss where it wants it. :meth:`LinkSession.drop_frame` loses
the next frame before the driver applies it, :meth:`LinkSession.drop_answer`
lets the driver apply the frame and loses its answer, and
:meth:`LinkSession.disconnect` takes the transport down whole. Either loss
closes the read end the other side is waiting on, so both ends learn of it at
once, as a closed connection tells both. The fake clock stamps the driver's
applications, so a test reads the arrival order without waiting.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import anyio
from midojo_sdk.link import Link, LinkRecorder, LinkWorld, encode_frame
from midojo_sdk.recorder import EnvironmentRevision

from midojo.control_plane.link_dispatch import LinkDispatcher
from midojo.control_plane.service import ControlPlane


class FakeClock:
    """A clock the test advances by hand: nothing here waits for real time."""

    def __init__(self, now: float = 1_700_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@dataclass
class _Drop:
    """A count of writes, and the point past which the next one is lost."""

    after: int | None = None
    written: int = 0

    def arm(self, after: int) -> None:
        self.after, self.written = after, 0

    def loses(self) -> bool:
        """Whether the write being made now is the one to lose; it counts either way."""
        if self.after is None:
            return False
        self.written += 1
        return self.written > self.after


class _Losses:
    """Where a session's test injects a loss, counted in frames and in answers written."""

    def __init__(self, to_driver: Any, to_service: Any) -> None:
        self._to_driver = to_driver
        self._to_service = to_service
        self.frames = _Drop()
        self.answers = _Drop()

    async def write_frame(self, send: Any, data: str) -> None:
        """Write one frame; the losing write closes its send stream first, so the driver reads the end of the stream."""
        if self.frames.loses():
            self._to_driver[0].close()
        await send(data)

    async def write_answer(self, send: Any, data: str) -> None:
        """Write one answer; the losing write closes its send stream first, so the service reads the end of the stream."""
        if self.answers.loses():
            self._to_service[0].close()
        await send(data)


class Duplex:
    """One end of the frame stream: an anyio send stream and a receive stream."""

    def __init__(self, send: Any, receive: Any) -> None:
        self.send_stream = send
        self.receive_stream = receive

    async def send(self, data: str) -> None:
        await self.send_stream.send(data)

    async def receive(self) -> str:
        return await self.receive_stream.receive()

    def close(self) -> None:
        """Close both directions, as a transport that went away does."""
        for stream in (self.send_stream, self.receive_stream):
            with suppress(Exception):
                stream.close()


class _ServiceEnd(Duplex):
    """The service's end, where a test's ``drop_frame`` loses a frame on its way out."""

    def __init__(self, send: Any, receive: Any, losses: _Losses) -> None:
        super().__init__(send, receive)
        self._losses = losses

    async def send(self, data: str) -> None:
        await self._losses.write_frame(self.send_stream.send, data)


class _DriverEnd(Duplex):
    """The driver's end, where a test's ``drop_answer`` loses an answer on its way back."""

    def __init__(self, send: Any, receive: Any, losses: _Losses) -> None:
        super().__init__(send, receive)
        self._losses = losses

    async def send(self, data: str) -> None:
        await self._losses.write_answer(self.send_stream.send, data)


class LinkSession:
    """One binding, wired end to end in memory: the service's link and the driver's dispatcher."""

    def __init__(
        self,
        *,
        link: Link,
        recorder: LinkRecorder,
        world: LinkWorld,
        dispatcher: LinkDispatcher,
        service_end: _ServiceEnd,
        driver_end: _DriverEnd,
        losses: _Losses,
        clock: FakeClock,
        serving: asyncio.Task[None],
    ) -> None:
        self.link = link
        self.recorder = recorder
        self.world = world
        self.dispatcher = dispatcher
        self.clock = clock
        self._service_end = service_end
        self._driver_end = driver_end
        self._losses = losses
        self._serving = serving

    @classmethod
    async def start(cls, control: ControlPlane, session_token: str, *, clock: FakeClock | None = None) -> LinkSession:
        """A live binding for ``session_token``: frames start flowing at once."""
        # A rendezvous stream: a frame is handed over, never queued, so a lost end is felt at once.
        to_driver = anyio.create_memory_object_stream[str](0)
        to_service = anyio.create_memory_object_stream[str](0)
        losses = _Losses(to_driver, to_service)
        service_end = _ServiceEnd(to_driver[0], to_service[1], losses)
        driver_end = _DriverEnd(to_service[0], to_driver[1], losses)
        clock = clock or FakeClock()
        dispatcher = LinkDispatcher(control, session_token, clock=clock)
        link = Link(service_end)
        serving = asyncio.get_running_loop().create_task(dispatcher.serve(driver_end))
        return cls(
            link=link,
            recorder=LinkRecorder(link),
            world=LinkWorld(link),
            dispatcher=dispatcher,
            service_end=service_end,
            driver_end=driver_end,
            losses=losses,
            clock=clock,
            serving=serving,
        )

    def drop_frame(self, after: int = 0) -> None:
        """Lose the frame after the first ``after`` frames: the driver applies nothing, and both ends lose the link."""
        self._losses.frames.arm(after)

    def drop_answer(self, after: int = 0) -> None:
        """Lose the answer after the first ``after`` answers: the driver applied its frame, and both ends lose the link."""
        self._losses.answers.arm(after)

    def disconnect(self) -> None:
        """End the transport whole: both directions close, and both ends read the loss."""
        self._service_end.close()
        self._driver_end.close()

    async def feed(self, frame: Any) -> None:
        """Write ``frame`` to the driver without waiting for an answer, as a stream that skipped an id does."""
        await self._service_end.send(encode_frame(frame))

    async def wait(self) -> None:
        """Wait for the driver's loop to end, as a test that showed it a frame it can't apply needs."""
        with suppress(asyncio.CancelledError):
            await self._serving

    async def aclose(self) -> None:
        """Stop the binding: the link ends, the dispatcher stops, the streams close."""
        await self.link.aclose()
        if not self._serving.done():
            self._serving.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await self._serving
        self._service_end.close()
        self._driver_end.close()


class LinkAgent:
    """The :class:`LinkRecorder` and :class:`LinkWorld` pair as one agent, started on first use.

    A ``midojo_sdk.recorder.Recorder`` (and ``DetailedRecorder`` and
    ``CaptureReporter``) and a ``WorldState``: the same surface as
    ``InProcessAgentClient``, over the link instead of a direct call.
    """

    def __init__(self, control: ControlPlane, session_token: str, *, clock: FakeClock | None = None) -> None:
        self.control = control
        self.session_token = session_token
        self.clock = clock or FakeClock()
        self.session: LinkSession | None = None

    async def session_now(self) -> LinkSession:
        """The live binding, starting it the first time a test or a call needs it."""
        if self.session is None:
            self.session = await LinkSession.start(self.control, self.session_token, clock=self.clock)
        return self.session

    async def start_call(self, **kwargs: Any) -> None:
        await (await self.session_now()).recorder.start_call(**kwargs)

    async def finish_call(self, **kwargs: Any) -> None:
        await (await self.session_now()).recorder.finish_call(**kwargs)

    async def finish_call_detailed(self, **kwargs: Any) -> None:
        await (await self.session_now()).recorder.finish_call_detailed(**kwargs)

    async def report_capture_failure(self, **kwargs: Any) -> None:
        await (await self.session_now()).recorder.report_capture_failure(**kwargs)

    async def read(self) -> EnvironmentRevision:
        return await (await self.session_now()).world.read()

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        return await (await self.session_now()).world.change(values, expected_revision=expected_revision)

    async def aclose(self) -> None:
        if self.session is not None:
            await self.session.aclose()
            self.session = None
