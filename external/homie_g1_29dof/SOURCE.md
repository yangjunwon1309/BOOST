## Upstream

This deployment snapshot is derived from the HOMIE project by Qingwei Ben,
Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin, and Jiangmiao Pang.

- Project: <https://github.com/InternRobotics/Homie>
- Compared deployment source: <https://github.com/InternRobotics/Homie/tree/main/HomieDeploy/unitree_sdk2>
- Local source: customized `homie_g1_29dof` deployment supplied for PATH-G1
- Revision: unknown; the supplied source folder had no Git metadata

## Local changes

This snapshot adapts the supplied deployment for PATH-G1, including its G1 DDS
control loop, arm target integration, configuration, and deployment guide. The
exact upstream commit and a complete file-by-file change history are unavailable.

## License

The HOMIE root README and `HomieDeploy/README.md` declare CC BY-NC-SA 4.0 for
HOMIE code. The upstream `HomieDeploy/LICENSE` currently contains CC BY-NC 4.0,
so the upstream notices are inconsistent. This PATH-G1 snapshot follows the
CC BY-NC-SA 4.0 declaration in the two README files; see the scoped
[`LICENSE`](LICENSE) notice. Third-party components retain their own licenses.
