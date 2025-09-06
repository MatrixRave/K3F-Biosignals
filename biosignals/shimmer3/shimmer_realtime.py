from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
import time

class Shimmer3RealtimeDataStream:

    def __init__(self, serial_port='/dev/tty.Shimmer3-A66F', baudrate=DEFAULT_BAUDRATE):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.serial = None
        self.shim_dev = None
        self.is_streaming = False
        self.data_callback_handler = None

    def initialize(self):
        try:
            self.serial = Serial(self.serial_port, self.baudrate)
            self.shim_dev = ShimmerBluetooth(self.serial)
            self.shim_dev.initialize()
            device_name = self.shim_dev.get_device_name()
            print(f'Connect to device: {device_name}')

            self.shim_dev.add_stream_callback(self.incoming_data_handler)

        except Exception as e:
            print(f"Fehler bei der Initialisierung: {e}")
            raise

    def set_callback_handler(self, callback_handler):
        self.data_callback_handler = callback_handler

    def incoming_data_handler(self, packet):
        if self.data_callback_handler is not None:
            self.data_callback_handler(packet)

    def start_streaming(self):
        if self.shim_dev is None:
            raise RuntimeError("Device was not initialized")
        try:
            self.shim_dev.start_streaming()
            self.is_streaming = True
            print("Starting streaming from shimmer device...")
        except Exception as e:
            print(f"Starting streaming failed: {e}")

    def stop_streaming(self):
        self.shim_dev.stop_streaming()
        self.is_streaming = False
        print("Stop streaming")

    def collect_data(self, collection_duration=10.0):
        self.start_streaming()
        try:
            time.sleep(collection_duration)
        except KeyboardInterrupt:
            print("Interrupted collection.")
        finally:
            self.stop_streaming()

    def shutdown(self):
        if self.is_streaming:
            self.stop_streaming()

        if self.shim_dev:
            try:
                self.shim_dev.shutdown()
                print("Stop reading thread")
            except Exception as e:
                print(f"Stopping reading thread failed: {e}")

        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Closed serial connection")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()