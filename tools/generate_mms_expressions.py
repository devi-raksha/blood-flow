#!/usr/bin/env python3
"""Derive manufactured sources for the implemented 1-D blood-flow equations.

The volume equations assembled by ``MetricFlowSystem`` are

  A_t + (A U)_x = f_A
  U_t + (U^2/2 + p(A)/rho)_x - eta U/A = f_U,

where ``eta = 2*(xi + 2)*pi*mu/rho`` and
``p(A) = p0 + pd + beta/ad*(sqrt(A) - sqrt(ad))`` with
``beta = 4*sqrt(pi)*E*h_wall/3``.  The script deliberately keeps vessel
quantities symbolic: those values are supplied by the VTK vessel record at
runtime, not by the current FunctionParser constant map.

The emitted expressions use deal.II FunctionParser syntax (``^`` for powers,
``PI`` for pi, and semicolon-separated vector components).  The generated
expressions are therefore suitable as a source artifact, but a clean runnable
case still requires exposing the vessel-specific constants to FunctionParser.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, Iterable

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover - exercised by CLI environments
    sp = None
    _SYMPY_ERROR = exc


@dataclass(frozen=True)
class MMSProfile:
    name: str
    area: str
    velocity: str


# These are the three exact functions used by the historical p1/p2/p3
# convergence cases in parameters/working_parameters.  Keep their strings
# explicit so that the source of the benchmark functions is auditable.
PROFILES = {
    "p1": MMSProfile(
        "p1",
        "0.1*a0*sin(2*PI*x)*cos(2*PI*t/5) + a0",
        "-0.02*a0*sin(2*PI*t/5)*cos(2*PI*x)",
    ),
    "p2": MMSProfile(
        "p2",
        "t*sin(2*PI*x) + 4",
        "cos(2*PI*x)/(2*PI*(t*sin(2*PI*x) + 4))",
    ),
    "p3": MMSProfile(
        "p3",
        "1 - 0.005*sin(9.42*t - 6.28*x)",
        "0.05*sin(6.28*x)",
    ),
}


def _require_sympy():
    if sp is None:
        raise RuntimeError(
            "SymPy is required to derive MMS expressions; install sympy first"
        ) from _SYMPY_ERROR


def _parser(expr) -> str:
    """Render a SymPy expression using deal.II FunctionParser spelling."""
    text = sp.sstr(sp.factor(expr))
    return text.replace("**", "^").replace("pi", "PI")


def derive_profile(profile: MMSProfile) -> Dict[str, str]:
    """Return symbolic fields, derivatives, and manufactured sources."""
    _require_sympy()
    x, t = sp.symbols("x t")
    a0, rho, mu, xi = sp.symbols("a0 rho mu xi")
    E, h_wall, a_d, p0, p_d = sp.symbols("E h_wall a_d p0 p_d")
    A = sp.sympify(profile.area.replace("PI", "pi"), locals={"pi": sp.pi})
    U = sp.sympify(profile.velocity.replace("PI", "pi"), locals={"pi": sp.pi})
    # sympify creates x/t and parameter symbols by name; replace with the
    # symbols above so differentiation is unambiguous.
    local = {str(s): s for s in (x, t, a0, rho, mu, xi, E, h_wall, a_d, p0, p_d)}
    A = sp.sympify(profile.area.replace("PI", "pi"), locals=local | {"pi": sp.pi})
    U = sp.sympify(profile.velocity.replace("PI", "pi"), locals=local | {"pi": sp.pi})
    beta = sp.Rational(4, 3) * sp.sqrt(sp.pi) * E * h_wall
    pressure = p0 + p_d + beta / a_d * (sp.sqrt(A) - sp.sqrt(a_d))
    eta = 2 * (xi + 2) * sp.pi * mu / rho
    mass = sp.factor(sp.diff(A, t) + sp.diff(A * U, x))
    momentum = sp.factor(
        sp.diff(U, t) + sp.diff(U**2 / 2 + pressure / rho, x) - eta * U / A
    )
    return {
        "area": _parser(A),
        "velocity": _parser(U),
        "area_t": _parser(sp.diff(A, t)),
        "area_x": _parser(sp.diff(A, x)),
        "velocity_t": _parser(sp.diff(U, t)),
        "velocity_x": _parser(sp.diff(U, x)),
        "mass_source": _parser(mass),
        "momentum_source": _parser(momentum),
        "rhs_expression": f"{_parser(mass)}; {_parser(momentum)}",
        "pressure_law": "p0 + p_d + (4/3)*sqrt(PI)*E*h_wall/a_d*(sqrt(A)-sqrt(a_d))",
        "eta": "2*(xi+2)*PI*mu/rho",
    }


def derive_all(names: Iterable[str] = PROFILES) -> Dict[str, Dict[str, str]]:
    return {name: derive_profile(PROFILES[name]) for name in names}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["all", *PROFILES], default="all")
    parser.add_argument("--output", type=argparse.FileType("w"), default="-")
    args = parser.parse_args(argv)
    try:
        names = PROFILES if args.profile == "all" else [args.profile]
        payload = {
            "schema": "blood-flow-mms-v1",
            "equations": "implemented volume equations in source/blood_flow_system.cc",
            "profiles": derive_all(names),
        }
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    json.dump(payload, args.output, indent=2, sort_keys=True)
    args.output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
