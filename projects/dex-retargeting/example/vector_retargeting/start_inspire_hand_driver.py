import time
from typing import Optional

import tyro
from loguru import logger

from dex_retargeting.constants import HandType


def main(
    hand_type: HandType,
    ip: Optional[str] = None,
    port: int = 6000,
    device_id: int = 1,
    network: Optional[str] = None,
    use_serial: bool = False,
    serial_port: str = "/dev/ttyUSB0",
    baudrate: int = 115200,
    state_poll_period: float = 0.05,
):
    """
    Starts a headless Inspire hand SDK driver that bridges DDS control topics to the physical hand.

    Args:
        hand_type: which hand topic to subscribe to.
        ip: Modbus TCP IP. If omitted, the SDK default IP is used.
        port: Modbus TCP port.
        device_id: Modbus device ID for the target hand.
        network: optional DDS network interface passed to ChannelFactoryInitialize.
        use_serial: switch the SDK driver to serial mode instead of TCP.
        serial_port: serial device path when use_serial is enabled.
        baudrate: serial baudrate when use_serial is enabled.
        state_poll_period: how often to poll and publish hand state; set to 0 to disable polling.
    """
    try:
        from inspire_sdkpy.inspire_sdk import ModbusDataHandler
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ImportError as exc:
        raise RuntimeError(
            "The Inspire hand driver requires inspire_sdkpy and unitree_sdk2py in the current environment."
        ) from exc

    if state_poll_period < 0:
        raise ValueError("state_poll_period must be non-negative.")

    if network is None:
        ChannelFactoryInitialize(0)
    else:
        ChannelFactoryInitialize(0, network)

    topic_suffix = "r" if hand_type is HandType.right else "l"
    driver = ModbusDataHandler(
        ip=ip,
        port=port,
        device_id=device_id,
        LR=topic_suffix,
        network=network,
        use_serial=use_serial,
        serial_port=serial_port,
        baudrate=baudrate,
        initDDS=False,
    )

    logger.info(
        "Inspire hand driver is ready on topic rt/inspire_hand/ctrl/{} (device_id={}, ip={}, serial={}).".format(
            topic_suffix,
            device_id,
            ip or "sdk-default",
            use_serial,
        )
    )

    try:
        while True:
            if state_poll_period > 0:
                driver.read()
                time.sleep(state_poll_period)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Inspire hand driver stopped by user.")


if __name__ == "__main__":
    tyro.cli(main)