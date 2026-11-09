"""Pin CycloneDDS to one network interface without the SDK's inline XML config.

`ChannelFactoryInitialize(id, networkInterface="eth0")` builds an XML configuration
string and hands it to the native library (unitree_sdk2py/core/channel.py, the
`ChannelConfigHasInterface` branch). On this ARM64 box that path aborts the process:

    *** buffer overflow detected ***: terminated
    Aborted (core dumped)

...but only inside an activated conda env. `etc/conda/activate.d/cyclonedds.sh` puts
a locally built CycloneDDS first on LD_LIBRARY_PATH, so the process loads a different
libddsc than the one the Python bindings expect, and the inline-config path overflows
inside it before any topic is opened. Outside the env the same call is fine, which is
what makes this look like a script bug rather than an environment one.

channel.py already knows about this -- it skips the inline config whenever
CYCLONEDDS_URI is set -- so the fix is to say the same thing through the environment
variable CycloneDDS itself parses, and pass networkInterface=None.
"""

import os

CONFIG_TEMPLATE = (
    "<CycloneDDS><Domain><General><Interfaces>"
    '<NetworkInterface name="{interface}"/>'
    "</Interfaces></General></Domain></CycloneDDS>"
)


def pin_interface(interface, logger=None):
    """Point CycloneDDS at `interface`. No-op when it is falsy or already configured.

    Must run before the first ChannelFactoryInitialize: the domain is a singleton and
    reads its configuration once.
    """
    if not interface:
        return False
    existing = os.environ.get("CYCLONEDDS_URI")
    if existing:
        if logger:
            logger.warning(f"CYCLONEDDS_URI is already set, leaving it alone: {existing}")
        return False
    os.environ["CYCLONEDDS_URI"] = CONFIG_TEMPLATE.format(interface=interface)
    if logger:
        logger.info(f"pinned CycloneDDS to {interface} via CYCLONEDDS_URI")
    return True
