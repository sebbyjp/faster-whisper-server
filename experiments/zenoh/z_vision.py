# import argparse
# import json
# import time

# import cv2
# import imutils
# from imutils.video import VideoStream
# import zenoh

# CAMERA_ID = 0

# parser = argparse.ArgumentParser(prog="zcapture", description="zenoh video capture example")
# parser.add_argument("-m", "--mode", type=str, choices=["peer", "client"], help="The zenoh session mode.")
# parser.add_argument(
#     "-e", "--connect", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on."
# )
# parser.add_argument(
#     "-l", "--listen", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on."
# )
# parser.add_argument("-w", "--width", type=int, default=500, help="width of the published frames")
# parser.add_argument("-q", "--quality", type=int, default=95, help="quality of the published frames (0 - 100)")
# parser.add_argument("-d", "--delay", type=float, default=0.05, help="delay between each frame in seconds")
# parser.add_argument("-k", "--key", type=str, default="/demo/zcam", help="key expression")
# parser.add_argument("-c", "--config", type=str, metavar="FILE", help="A zenoh configuration file.")

# args = parser.parse_args()

# jpeg_opts = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]

# conf = zenoh.config_from_file(args.config) if args.config is not None else zenoh.Config()
# if args.mode is not None:
#     conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(args.mode))
# if args.connect is not None:
#     conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
# if args.listen is not None:
#     conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(args.listen))


# print("[INFO] Open zenoh session...")
# zenoh.init_logger()
# z = zenoh.open(conf)

# print("[INFO] Open camera...")
# vs = VideoStream(src=CAMERA_ID).start()

# time.sleep(1.0)

# while True:
#     raw = vs.read()
#     if raw is not None:
#         frame = imutils.resize(raw, width=args.width)
#         _, jpeg = cv2.imencode(".jpg", frame, jpeg_opts)
#         z.put(args.key, jpeg.tobytes())

#     time.sleep(args.delay)


import argparse
import json
import time

import cv2
import numpy as np
import zenoh

parser = argparse.ArgumentParser(prog="zdisplay", description="zenoh video display example")
parser.add_argument("-m", "--mode", type=str, choices=["peer", "client"], help="The zenoh session mode.")
parser.add_argument(
    "-e", "--connect", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on."
)
parser.add_argument(
    "-l", "--listen", type=str, metavar="ENDPOINT", action="append", help="zenoh endpoints to listen on.", 
    default=["udp//0.0.0.0:3454"],
)
parser.add_argument("-d", "--delay", type=float, default=0.05, help="delay between each frame in seconds")
parser.add_argument("-k", "--key", type=str, default="demo/zcam", help="key expression")
parser.add_argument("-c", "--config", type=str, metavar="FILE", help="A zenoh configuration file.")

args = parser.parse_args()

conf = zenoh.Config.from_file(args.config) if args.config is not None else zenoh.Config()
if args.mode is not None:
    conf.insert_json5(zenoh.config.MODE_KEY, json.dumps(args.mode))
if args.connect is not None:
    conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps(args.connect))
if args.listen is not None:
    conf.insert_json5(zenoh.config.LISTEN_KEY, json.dumps(args.listen))

cams = {}


def frames_listener(sample):
    npImage = np.frombuffer(bytes(sample.value.payload), dtype=np.uint8)
    matImage = cv2.imdecode(npImage, 1)

    cams[sample.key_expr] = matImage


print("[INFO] Open zenoh session...")
zenoh.init_logger()
z = zenoh.open(conf)

sub = z.declare_subscriber(args.key, frames_listener)

while True:
    for cam in list(cams):
        cv2.imshow(str(cam), cams[cam])

    key = cv2.waitKey(1) & 0xFF
    time.sleep(args.delay)
