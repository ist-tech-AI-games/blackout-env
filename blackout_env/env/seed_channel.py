import uuid

from mlagents_envs.side_channel.side_channel import SideChannel, OutgoingMessage, IncomingMessage


class SeedChannel(SideChannel):
    """
    One-way SideChannel that sends an integer seed from Python to Unity.

    Unity calls Random.InitState(seed) upon receiving the message,
    before OnEpisodeBegin runs, so map/item generation is reproducible.
    """

    CHANNEL_ID = uuid.UUID("7a8b9c0d-1e2f-3a4b-5c6d-7e8f9a0b1c2d")

    def __init__(self) -> None:
        super().__init__(self.CHANNEL_ID)

    def on_message_received(self, msg: IncomingMessage) -> None:
        pass  # Python → Unity only; no incoming messages expected

    def send_seed(self, seed: int) -> None:
        msg = OutgoingMessage()
        msg.write_int32(seed)
        self.queue_message_to_send(msg)
