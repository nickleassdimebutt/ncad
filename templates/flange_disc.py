# genloop template: circular flange — disc, center bore, bolt circle,
# optional hub boss. All scalars overridable via --params.
od = 80.0
thickness = 8.0
edge_chamfer = 0.0
flutes = 0.0
bore_d = 25.0
bolt_d = 6.0
bolt_n = 4
bcd = 55.0
hub_d = 0.0
hub_h = 0.0

part = Cylinder(od / 2, thickness)
if hub_d > 0 and hub_h > 0:
    part += Cylinder(hub_d / 2, hub_h).translate((0, 0, thickness / 2 + hub_h / 2))
if edge_chamfer > 0:
    part = chamfer(part.edges().group_by(Axis.Z)[-1], min(edge_chamfer, thickness * 0.45))
if flutes >= 3:                       # vertical rectangular reeding cut into the rim.
    # RECTANGULAR (planar) grooves, not cylinders: they don't read as bolt holes to
    # the feature checker, and the depth is capped to the rim wall so they can never
    # sever the ring or empty the disc.
    rim = (od - bore_d) / 2 if 0 < bore_d < od else od / 2
    gd = min(od * 0.06, rim * 0.5)
    if gd > 0.3:
        for loc in PolarLocations(od / 2, int(flutes)):
            part -= loc * Box(gd * 2, od * 0.05, (thickness + hub_h) * 2)
if 0 < bore_d < od:                   # an oversized bore would consume the whole disc
    part -= Cylinder(bore_d / 2, (thickness + hub_h) * 2.5)
if bolt_d > 0:
    for loc in PolarLocations(bcd / 2, int(bolt_n)):
        part -= loc * Cylinder(bolt_d / 2, thickness * 2)

show_object(part)
