# genloop template: rectangular plate with optional fillets, center bore,
# and a polar or grid hole pattern. All scalars overridable via --params.
length = 60.0
width = 40.0
thickness = 5.0
corner_fillet = 0.0
edge_chamfer = 0.0
pockets = 0.0
bore_d = 0.0
hole_d = 0.0
hole_n = 4
pattern = 1        # 1 = polar bolt circle, 0 = grid
circle_d = 40.0
grid_dx = 20.0
grid_dy = 20.0
grid_nx = 2
grid_ny = 2

part = Box(length, width, thickness)
if corner_fillet > 0:
    part = fillet(part.edges().filter_by(Axis.Z), corner_fillet)
if edge_chamfer > 0:
    part = chamfer(part.edges().group_by(Axis.Z)[-1], min(edge_chamfer, thickness * 0.45))
if pockets >= 1:                      # rectangular lightening pockets in the top face
    pw, ph, pd = length * 0.28, width * 0.28, thickness * 0.45
    for loc in GridLocations(length * 0.42, width * 0.42, 2, 2):
        part -= loc * Box(pw, ph, pd * 2).translate((0, 0, thickness / 2))
if bore_d > 0:
    part -= Cylinder(bore_d / 2, thickness * 2)
if hole_d > 0:
    if pattern >= 1:
        # keep the bolt circle fully inside the plate face
        circle_d = min(circle_d, min(length, width) - hole_d - 2.0)
        for loc in PolarLocations(circle_d / 2, int(hole_n)):
            part -= loc * Cylinder(hole_d / 2, thickness * 2)
    else:
        for loc in GridLocations(grid_dx, grid_dy, int(grid_nx), int(grid_ny)):
            part -= loc * Cylinder(hole_d / 2, thickness * 2)

show_object(part)
