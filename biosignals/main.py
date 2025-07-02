from driver_vision_processor import DriverVisionProcessor


def main():
    processor = DriverVisionProcessor(video_source=0)
    processor.start_video_analysis()



if __name__ == '__main__':
    main()