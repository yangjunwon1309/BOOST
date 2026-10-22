# MuJoCo scene assets

## Table room

The room XML and visual assets originate from
[k-valentin/unitree-g1-mujoco-worlds, `worlds/table_room_1x3450`](https://huggingface.co/k-valentin/unitree-g1-mujoco-worlds/tree/051f752a119d4ded1f05d8b0d33bc8790012cba4/worlds/table_room_1x3450).
`table_room_1x3450/world_upstream.xml` and `world_upstream.yaml` retain the
upstream files. The local `world.xml` is an include fragment: it relocates the
asset paths and inlines the upstream geometry defaults. It retains the source
Pringles can and its collision cylinder, positioned alongside the articulated
box. Table, partition, and can meshes are under
`mujoco/meshes/table_room_1x3450/`; room textures are beside the room XML. The
upstream YCB license notice is retained as `table_room_1x3450/YCB_LICENSE.txt`.

## Active articulated cardboard box

The active box model and manifest originate from
[projectsim/articulated-boxes, `cardboard_box_0002`](https://huggingface.co/datasets/projectsim/articulated-boxes/tree/bbfcf129f30a9c19f4e17c3c27bf1460d0346b23/boxes/cardboard_box_0002).
The dataset is attributed to projectsim by Kaedim and declares CC BY 4.0.
`cardboard_box_0002_source.xml` and `manifest.json` are retained. The local
`cardboard_box_0002.xml` preserves its five rigid links and four flap hinges,
converts hinge limits to radians for the G1 scene's compiler, namespaces its
MuJoCo object names, sets zero collision margin for surface queries, and places
it on the table.

The previous `cardboard_box_0001` model, source XML, and manifest remain in
`articulated_boxes/cardboard_box_0001/` unchanged.
