import time
from serial import Serial
from pyshimmer import ShimmerBluetooth, DataPacket, EChannelType


def handler(pkt: DataPacket) -> None:
    print("GSR_RAW", pkt[EChannelType.GSR_RAW])  # Raw GSR Value
    print("PPG", pkt[EChannelType.INTERNAL_ADC_13]) # Raw PPG Value
    print("VBATT", pkt[EChannelType.VBATT]) # Battery Status


if __name__ == '__main__':
    com_port = '/dev/tty.Shimmer3-A66F' # Bluetooth-Device (pairing)
    serial = Serial(com_port, 115200)
    shimmer3_device = ShimmerBluetooth(serial)  # API

    shimmer3_device.initialize()  # Initialize Bluetooth connection

    dev_name = shimmer3_device.get_device_name()
    print(f'Device name is: {dev_name}')

    # Add a stream callback which is called when a new data packet arrives
    shimmer3_device.add_stream_callback(handler)

    # Start listening for data stream for 60s
    shimmer3_device.start_streaming()
    time.sleep(60.0)
    shimmer3_device.stop_streaming()

    # Shutdown the read loop
    shimmer3_device.shutdown()