"""Initialize camera CPU affinity before loading camera/native dependencies."""

from .cpu_affinity import bootstrap_camera_affinity


def main(argv=None):
    bootstrap_camera_affinity(argv)
    from .image_server import main as run_server
    return run_server(argv)


if __name__ == '__main__':
    main()
