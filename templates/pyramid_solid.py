# nCAD template: solid rectangular pyramid with a centered apex.
# All scalars are overridable via --params.
base_x = 100.0
base_y = 100.0
height = 80.0

z0 = -height / 2
base = Wire.make_polygon([
    (-base_x / 2, -base_y / 2, z0),
    ( base_x / 2, -base_y / 2, z0),
    ( base_x / 2,  base_y / 2, z0),
    (-base_x / 2,  base_y / 2, z0),
], close=True)
part = Solid.make_loft([base, Vertex(0, 0, height / 2)], ruled=True)

show_object(part)
