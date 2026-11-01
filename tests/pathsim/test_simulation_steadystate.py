"""Coupled steady-state equilibria and continuation after a failed solve."""

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import (
    Constant, Adder, Amplifier, Integrator, Scope, ODE, DynamicalSystem, PID
    )
from pathsim.solvers import RKCK54
from pathsim.subsystem import Subsystem, Interface


class TestCoupledSteadyState(unittest.TestCase):

    def closed_loop(self, target=1.0, Ki=2.0, Kd=0.0, nested=False):
        ref, err, ctrl, scope = Constant(target), Adder("+-"), PID(1., Ki, Kd, 50), Scope()
        plant = DynamicalSystem(
            func_dyn=lambda x, u, t: -x**2 + u,
            func_alg=lambda x, u, t: x,
            initial_value=1.0,
            jac_dyn=lambda x, u, t: -2*x
            )
        wrapped = plant
        if nested:
            for _ in range(2):
                interface = Interface()
                wrapped = Subsystem(
                    [interface, wrapped],
                    [Connection(interface, wrapped), Connection(wrapped, interface)]
                    )
        sim = Simulation(
            [ref, err, ctrl, wrapped, scope],
            [Connection(ref, err), Connection(wrapped, err[1], scope),
             Connection(err, ctrl), Connection(ctrl, wrapped)],
            Solver=RKCK54, log=False
            )
        return sim, ctrl, plant, scope

    def test_pid_equilibrium(self):
        #integral action enforces x=target; then Ki*z=x**2
        for target, Ki, Kd in [(1., 2., 0.), (.5, .5, .1), (2., 3., .2)]:
            with self.subTest(target=target, Ki=Ki, Kd=Kd):
                sim, ctrl, plant, scope = self.closed_loop(target, Ki, Kd)
                sim.steadystate()
                np.testing.assert_allclose(ctrl.state, [0., target**2/Ki], atol=1e-9)
                np.testing.assert_allclose(plant.state, [target], atol=1e-9)
                np.testing.assert_allclose(ctrl.derivative(0), 0., atol=sim.tolerance_fpi)
                np.testing.assert_allclose(plant.derivative(0), 0., atol=sim.tolerance_fpi)
                self.assertEqual(sim.Solver, RKCK54)
                self.assertEqual(len(scope.recording_time), 1)

    def test_nested_equilibrium(self):
        #a subsystem's dummy engine is not its dynamic state
        sim, ctrl, plant, _ = self.closed_loop(nested=True)
        sim.steadystate()
        np.testing.assert_allclose(ctrl.state, [0., .5], atol=1e-9)
        np.testing.assert_allclose(plant.state, [1.], atol=1e-9)

    def test_equilibrium_then_transient(self):
        sim, ctrl, plant, _ = self.closed_loop()
        sim.steadystate()
        sim.run(.1, reset=False)
        np.testing.assert_allclose(ctrl.state, [0., .5], atol=1e-9)
        np.testing.assert_allclose(plant.state, [1.], atol=1e-9)

    def test_algebraic_diagram(self):
        source, gain, scope = Constant(3.), Amplifier(2.), Scope()
        sim = Simulation([source, gain, scope], [Connection(source, gain), Connection(gain, scope)], log=False)
        sim.steadystate()
        self.assertEqual(gain.outputs[0], 6.)
        self.assertEqual(len(scope.recording_time), 1)

    def test_nonconvergence_restores_solver(self):
        #dx/dt=1 has no equilibrium, but a failed solve must allow a transient
        source, integ, scope = Constant(1.), Integrator(), Scope()
        sim = Simulation([source, integ, scope], [Connection(source, integ), Connection(integ, scope)],
                         Solver=RKCK54, iterations_max=3, log=False)
        with self.assertRaisesRegex(RuntimeError, "STEADYSTATE -> FAILED"):
            sim.steadystate()
        self.assertEqual(sim.Solver, RKCK54)
        self.assertIsInstance(integ.engine, RKCK54)
        self.assertEqual(len(scope.recording_time), 0)
        sim.run(.1, reset=False)
        np.testing.assert_allclose(integ.state, [.1], atol=1e-10)

    def test_rhs_exception_restores_solver(self):
        def invalid_rhs(x, u, t):
            raise ValueError("invalid model")

        block = ODE(invalid_rhs)
        sim = Simulation([block], Solver=RKCK54, log=False)
        with self.assertRaisesRegex(ValueError, "invalid model"):
            sim.steadystate()
        self.assertEqual(sim.Solver, RKCK54)
        self.assertIsInstance(block.engine, RKCK54)

    def test_inactive_state_is_unchanged(self):
        sim, _, plant, _ = self.closed_loop()
        inactive = Integrator(7.)
        sim.add_block(inactive)
        inactive.off()
        sim.steadystate()
        np.testing.assert_allclose(plant.state, [1.], atol=1e-9)
        np.testing.assert_array_equal(inactive.state, [7.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
