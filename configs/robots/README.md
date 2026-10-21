# Robot configuration

The target location for the G1 joint map and wrist-camera extrinsics is this
directory. The imported PATH runtime still reads its camera calibration from
`services/pose_service/config/`; move it here when the BOOST robot config
schema is introduced, preserving the current wrist-yaw-link frame convention.
