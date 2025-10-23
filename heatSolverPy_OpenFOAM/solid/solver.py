from heat2d import Heat2d


def main(time, couplingMethod, testing=True):
    solver = Heat2d(time, couplingMethod)
    solver.setLeftBoundaryCondition('temp', 273.15)
    solver.setRightBoundaryCondition('temp', 274.15)
    solver.setColorMapScale(273, 274.2)
    i = 0
    while i < solver.time:
        solver.calculateHeatEquation(i, animate=False)
        print("Python Time: {:f}".format(i))
        i += solver.dt
    if not testing:
        solver.plotTemperature()
    
    return solver.getInterfaceTemperature()

if __name__ == "__main__":
    main(time=70, couplingMethod="linearInterpolation", testing=False)
