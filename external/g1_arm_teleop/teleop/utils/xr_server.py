"""Check XR server availability before creating robot command publishers."""

import errno
import socket


def ensure_xr_port_available(port=8012):
    """Reject a second XR session; leave the existing controller running."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('0.0.0.0', port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"XR port {port} is already in use. Stop the previous teleop with Ctrl+C "
                    "and wait for it to exit before starting another session. The headset "
                    "would otherwise connect to the previous XR server; robot control "
                    "has not been started by this process."
                ) from error
            raise
