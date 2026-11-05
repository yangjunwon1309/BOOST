"""Small camera configuration helpers without server dependencies."""


def video_path_for_id(video_id):
    if video_id is None:
        return None
    return f"/dev/video{video_id}"
