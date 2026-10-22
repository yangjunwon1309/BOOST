"""Local ZMQ wire fields for MuJoCo fingertip surface queries."""

SURFACE_QUERY_TOPIC = "mujoco_surface_query"
SURFACE_PROTOCOL_VERSION = 1
SIDE_TO_ID = {"left": 0, "right": 1}
ID_TO_SIDE = {value: key for key, value in SIDE_TO_ID.items()}
NUM_TIPS = 5
