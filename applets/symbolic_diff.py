# standard library
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict

# sympy
import sympy as sp
from sympy import symbols, diff, lambdify, sqrt
from sympy import Matrix, Symbol
from sympy.core.expr import Expr

# numpy
import numpy as np

# pandas
import pandas as pd

# deep-signature-2
from core import utils as core_utils
from data.generation import GaussianPatchGenerator, InverseFourierPatchGenerator, RBFPatchGenerator, SimplexNoisePatchGenerator, QuadraticMonagePatchGenerator, QuadraticMonagePatchGenerator2
from data.evaluation import CorrelationEvaluator


def compute_grad(f: Expr, u: Symbol, v: Symbol) -> Matrix:
    # Compute the gradient of f
    f_u = sp.diff(f, u)
    f_v = sp.diff(f, v)
    return sp.Matrix([f_u, f_v])


def compute_weingarten_matrix(x: Matrix, u: Symbol, v: Symbol) -> Matrix:
    # Compute the tangent vectors
    x_u = x.diff(u)
    x_v = x.diff(v)

    # Compute the normal vector
    n = x_u.cross(x_v)
    n = n / n.norm()

    # Compute the second derivatives
    x_uu = x_u.diff(u)
    x_vv = x_v.diff(v)
    x_uv = x_u.diff(v)

    # Compute Weingarten matrix
    w = sp.Matrix([
        [-x_uu.dot(n), -x_uv.dot(n)],
        [-x_uv.dot(n), -x_vv.dot(n)]
    ])

    return w


def compute_principal_directions(w: Matrix) -> tuple[Matrix, Matrix]:
    # Compute eigenvectors
    eigenvectors = w.eigenvects()

    # The principal directions are the eigenvectors
    d1 = eigenvectors[0][2][0]
    d2 = eigenvectors[1][2][0]

    return d1, d2


def substitute_point_into_expression(f: Expr, point: Dict[Symbol, float]) -> Expr:
    # Substitute the point into the derivative
    expr = f.subs(point)
    return expr


def main():
    L, N, A, B, C, u, v = symbols('L N A B C u v')
    h = (L / 2) * (u ** 2) + (N / 2) * (v ** 2) + A * u * v
    # h = (L / 2) * (u ** 2) + (N / 2) * (v ** 2)

    x = sp.Matrix([u, v, h])
    h_u = diff(h, u)
    h_v = diff(h, v)
    h_uu = diff(h_u, u)
    h_uv = diff(h_u, v)
    h_vv = diff(h_v, v)

    K = (h_uu * h_vv - h_uv ** 2) / ((1 + h_u ** 2 + h_v ** 2) ** 2)
    H = ((1 + h_v ** 2) * h_uu - 2 * h_u * h_v * h_uv + (1 + h_u ** 2) * h_vv) / (2 * ((1 + h_u ** 2 + h_v ** 2) ** (3/2)))
    k1 = H + sqrt(H ** 2 - K)
    k2 = H - sqrt(H ** 2 - K)

    w = compute_weingarten_matrix(x=x, u=u, v=v)
    d1, d2 = compute_principal_directions(w=w)
    print(f'w: {w}')
    print(f'd1: {d1}')
    print(f'd2: {d2}')

    grad_k1 = compute_grad(f=k1, u=u, v=v)
    grad_k2 = compute_grad(f=k2, u=u, v=v)

    # Compute the derivative in the principal directions
    k1_1 = grad_k1.dot(d1)
    k1_2 = grad_k1.dot(d2)
    k2_1 = grad_k2.dot(d1)
    k2_2 = grad_k2.dot(d2)

    grad_k1_2 = compute_grad(f=k1_2, u=u, v=v)
    grad_k2_1 = compute_grad(f=k2_1, u=u, v=v)

    k1_22 = grad_k1_2.dot(d2)
    k2_11 = grad_k2_1.dot(d1)

    # point = {
    #     u: 2.0,
    #     v: 1.0
    # }
    #
    # k1_eval = substitute_point_into_expression(f=k1, point=point).doit()
    # k2_eval = substitute_point_into_expression(f=k2, point=point).doit()
    # k1_1_eval = substitute_point_into_expression(f=k1_1, point=point).doit()
    # k1_2_eval = substitute_point_into_expression(f=k1_2, point=point).doit()
    # k2_1_eval = substitute_point_into_expression(f=k2_1, point=point).doit()
    # k2_2_eval = substitute_point_into_expression(f=k2_2, point=point).doit()
    # k1_22_eval = substitute_point_into_expression(f=k1_22, point=point).doit()
    # k2_11_eval = substitute_point_into_expression(f=k2_11, point=point).doit()

    print(f'H: {H}')
    print(f'K: {K}')
    print(f'k1: {k1}')
    print(f'k2: {k2}')
    # print(f'k1_1: {k1_1_eval}')
    # print(f'k1_2: {k1_2_eval}')
    # print(f'k2_1: {k2_1_eval}')
    # print(f'k2_2: {k2_2_eval}')
    # print(f'k1_22: {k1_22_eval}')
    # print(f'k2_11: {k2_11_eval}')

    # vars = (u, v, L, N, A, B)
    vars = (u, v, L, N, A)

    print(f'd1: {d1}')
    print(f'd2: {d2}')
    d1_numpy = lambdify(args=vars, expr=d1, modules="numpy")
    d2_numpy = lambdify(args=vars, expr=d2, modules="numpy")
    bla1 = d1_numpy(2, 1, -1, 1, 1)
    bla2 = d2_numpy(2, 1, -1, 1, 1)
    pass

    k1_numpy = lambdify(args=vars, expr=k1, modules="numpy")
    k2_numpy = lambdify(args=vars, expr=k2, modules="numpy")
    k1_1_numpy = lambdify(args=vars, expr=k1_1, modules="numpy")
    k1_2_numpy = lambdify(args=vars, expr=k1_2, modules="numpy")
    k2_1_numpy = lambdify(args=vars, expr=k2_1, modules="numpy")
    k2_2_numpy = lambdify(args=vars, expr=k2_2, modules="numpy")
    k1_22_numpy = lambdify(args=vars, expr=k1_22, modules="numpy")
    k2_11_numpy = lambdify(args=vars, expr=k2_11, modules="numpy")



    # k1_numpy = lambdify(args=vars, expr=k1_eval, modules="numpy")
    # k2_numpy = lambdify(args=vars, expr=k2_eval, modules="numpy")
    # k1_1_numpy = lambdify(args=vars, expr=k1_1_eval, modules="numpy")
    # k1_2_numpy = lambdify(args=vars, expr=k1_2_eval, modules="numpy")
    # k2_1_numpy = lambdify(args=vars, expr=k2_1_eval, modules="numpy")
    # k2_2_numpy = lambdify(args=vars, expr=k2_2_eval, modules="numpy")
    # k1_22_numpy = lambdify(args=vars, expr=k1_22_eval, modules="numpy")
    # k2_11_numpy = lambdify(args=vars, expr=k2_11_eval, modules="numpy")

    coeff_limit1 = 0.1
    coeff_limit2 = 1
    samples_count = 1000000
    rng = np.random.default_rng()
    coeffs1 = rng.uniform(low=-coeff_limit1, high=coeff_limit1, size=(samples_count, 2))
    coeffs2 = rng.uniform(low=-coeff_limit2, high=coeff_limit2, size=(samples_count, 4))
    # lambda_args = [coeffs1[:, 0], coeffs1[:, 1], coeffs2[:, 0], coeffs2[:, 1], coeffs2[:, 2], coeffs2[:, 3]]
    lambda_args = [coeffs1[:, 0], coeffs1[:, 1], coeffs2[:, 0], coeffs2[:, 1]]

    k1_vals = k1_numpy(*lambda_args)
    k2_vals = k2_numpy(*lambda_args)
    k1_1_vals = k1_1_numpy(*lambda_args)
    k1_2_vals = k1_2_numpy(*lambda_args)
    k2_1_vals = k2_1_numpy(*lambda_args)
    k2_2_vals = k2_2_numpy(*lambda_args)
    k1_22_vals = k1_22_numpy(*lambda_args)
    k2_11_vals = k2_11_numpy(*lambda_args)

    vals = np.stack([k1_vals, k2_vals, k1_1_vals, k1_2_vals, k2_1_vals, k2_2_vals, k1_22_vals, k2_11_vals])
    corr = np.corrcoef(vals)
    df = pd.DataFrame(corr)
    pd.set_option('display.precision', 5)
    print(df)
    df.to_excel(str(Path("./corr.xlsx")))


if __name__ == "__main__":
    main()
