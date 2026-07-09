# nCAD template: open-top enclosure shell — outer box minus cavity,
# optional outer corner fillets and floor bolt-circle holes.
outer_x = 100.0
outer_y = 60.0
outer_z = 30.0
wall = 2.5
floor = 2.5
corner_fillet = 3.0
hole_d = 0.0
hole_n = 4
circle_d = 40.0

part = Box(outer_x, outer_y, outer_z)
if corner_fillet > 0:
    part = fillet(part.edges().filter_by(Axis.Z), corner_fillet)
cavity = Box(outer_x - 2 * wall, outer_y - 2 * wall, outer_z - floor)
part -= cavity.translate((0, 0, floor / 2 + 0.005))
if hole_d > 0:
    # holes pierce the floor slab, which sits at the bottom of the outer box
    for loc in PolarLocations(circle_d / 2, int(hole_n)):
        part -= (loc * Cylinder(hole_d / 2, floor * 3)).translate((0, 0, -outer_z / 2 + floor / 2))

show_object(part)
