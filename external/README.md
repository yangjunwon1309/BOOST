# External Dependencies

PATH-G1 vendors the modified SONIC runtime because it is part of the PATH
system, but it does not vendor GMR or cuRoboV2.

Expected Thor host paths:

- `PATH_GROOT_ROOT=/home/url/GR00T-WholeBodyControl`
- `PATH_CUROBO_ROOT=/home/url/GR00T-WholeBodyControl/external_dependencies/curobo_v2`
- `PATH_GMR_ROOT=/home/url/GMR`
- `PATH_SONIC_DEPLOY_ROOT=/home/url/PATH-G1/sonic_runtime/gear_sonic_deploy`

Inside the deploy Docker, `~/PATH-G1` is mounted as `/workspace/PATH-G1`, so
`configs/runtime/local.env` resolves:

- `PATH_G1_ROOT=/workspace/PATH-G1`
- `PATH_SONIC_DEPLOY_ROOT=/workspace/PATH-G1/sonic_runtime/gear_sonic_deploy`

