from shimmer3 import shimmer_realtime
from pyshimmer import DataPacket, EChannelType

from shimmer3.shimmer_realtime import Shimmer3RealtimeDataStream



class ShimmerGSRplus:

    def __init__(self, serial_port='/dev/tty.Shimmer3-A66F'):
        if serial_port:
            self.device_stream = Shimmer3RealtimeDataStream(serial_port)


    def start_collection(self, collection_duration=10):
        with self.device_stream as collector:
            collector.initialize()
            collector.collect_data(collection_duration)


    def eda_callibration(self, raw_gsr_data):
        pass

    def convert_gsr_to_conductance(gsr_raw, gsr_range=0):
        pass