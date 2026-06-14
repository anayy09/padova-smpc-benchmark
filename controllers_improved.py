"""Controller implementations aligned with the manuscript."""

from __future__ import annotations

from typing import Optional, Dict, Tuple
import time

import numpy as np
import osqp
from scipy import sparse as sp

from digital_twin import StochasticDigitalTwin
import config


class StochasticMPC:
    """Chance-constrained MPC solved via a custom OSQP formulation."""

    def __init__(self, digital_twin: StochasticDigitalTwin):
        self.dt_model = digital_twin
        self.N = config.MPC_HORIZON
        self.n_states = digital_twin.n_states
        self.measurement_index = 1
        self.C_row = np.zeros(self.n_states)
        self.C_row[self.measurement_index] = 1.0

        self.w_glucose = config.W_GLUCOSE
        self.w_insulin = config.W_INSULIN
        self.w_delta_u = config.W_DELTA_U
        self.w_slack = config.W_RISK_SLACK
        self.w_low_glucose = config.W_LOW_GLUCOSE

        self.G_target = config.TARGET_GLUCOSE
        self.G_min = config.G_MIN
        self.G_max = config.G_MAX
        self.G_guard = config.G_GUARD
        self.kappa_hypo = config.KAPPA_HYPO
        self.kappa_hyper = config.KAPPA_HYPER

        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        self.delta_u_max = config.DELTA_U_MAX

        basal_u_hr = digital_twin.patient_params.get('basal_rate', 1.0)
        self.basal_u = basal_u_hr / 60.0
        self.last_u = self.basal_u

        self.osqp_solver = None
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "not_run"

    @staticmethod
    def _meal_vector(meal_cho: float, n_states: int) -> np.ndarray:
        vec = np.zeros(n_states)
        if meal_cho <= 0:
            return vec
        grams = meal_cho * 1000.0
        vec[6] += grams * 0.5
        vec[7] += grams * 0.5
        return vec

    def _predict_variances(self, state_cov: np.ndarray, F: np.ndarray) -> np.ndarray:
        P = state_cov.copy()
        variances = np.zeros(self.N)
        process_noise = self.dt_model.Q * self.dt_model.dt
        H = np.zeros(self.n_states)
        H[self.measurement_index] = 1.0
        for k in range(self.N):
            P = F @ P @ F.T + process_noise
            variances[k] = H @ P @ H + self.dt_model.R
        return variances

    def _prediction_matrices(
        self, state_mean: np.ndarray, F: np.ndarray, B: np.ndarray, meal_schedule: Dict[int, float]
    ) -> Tuple[np.ndarray, np.ndarray, str, int]:
        S = np.zeros((self.N + 1, self.n_states, self.N))
        v = np.zeros((self.N + 1, self.n_states))
        v[0] = state_mean
        for k in range(self.N):
            meal_vec = self._meal_vector(meal_schedule.get(k, 0.0), self.n_states)
            v[k] = v[k] + meal_vec
            S[k + 1] = F @ S[k]
            S[k + 1][:, k] += B[:, 0]
            v[k + 1] = F @ v[k]
        coeff = S[1:]
        const = v[1:]
        G = np.zeros((self.N, self.N))
        g0 = np.zeros(self.N)
        for k in range(self.N):
            G[k, :] = self.C_row @ coeff[k]
            g0[k] = self.C_row @ const[k]
        return G, g0

    def _build_difference_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        D = np.zeros((self.N, self.N))
        D[0, 0] = 1.0
        for k in range(1, self.N):
            D[k, k] = 1.0
            D[k, k - 1] = -1.0
        d_ref = np.zeros(self.N)
        d_ref[0] = self.last_u
        return D, d_ref

    def _solve_qp(
        self,
        G: np.ndarray,
        g0: np.ndarray,
        sigma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.N
        D, d_ref = self._build_difference_matrix()

        target_vec = np.full(n, self.G_target)
        basal_vec = np.full(n, self.basal_u)

        H_u = 2 * (
            self.w_glucose * (G.T @ G)
            + self.w_insulin * np.eye(n)
            + self.w_delta_u * (D.T @ D)
        ) + 1e-6 * np.eye(n)
        f_u = (
            2 * self.w_glucose * (G.T @ (g0 - target_vec))
            - 2 * self.w_insulin * basal_vec
            - 2 * self.w_delta_u * (D.T @ d_ref)
        )

        n_dec = 4 * n
        P = np.zeros((n_dec, n_dec))
        P[:n, :n] = H_u
        q = np.zeros(n_dec)
        q[:n] = f_u
        q[n:2 * n] = self.w_slack
        q[2 * n:3 * n] = self.w_slack
        q[3 * n:] = self.w_low_glucose

        # Constraint assembly
        A_blocks = []
        l_list = []
        u_list = []

        # 1. Control bounds
        A_bounds = np.hstack([np.eye(n), np.zeros((n, 3 * n))])
        A_blocks.append(A_bounds)
        l_list.append(np.full(n, self.u_min))
        u_list.append(np.full(n, self.u_max))

        # 2. Rate limits
        A_rate = np.hstack([D, np.zeros((n, 3 * n))])
        A_blocks.append(A_rate)
        l_list.append(-self.delta_u_max + d_ref)
        u_list.append(self.delta_u_max + d_ref)

        # 3. Hypoglycemia chance constraint
        A_hypo = np.hstack([-G, -np.eye(n), np.zeros((n, n)), np.zeros((n, n))])
        b_hypo = -(self.G_min - g0 + self.kappa_hypo * sigma)
        A_blocks.append(A_hypo)
        l_list.append(np.full(n, -np.inf))
        u_list.append(b_hypo)

        # 4. Low-glucose guardrail (soft)
        guard_rhs = self.G_guard - g0 + self.kappa_hypo * sigma
        A_guard = np.hstack([G, np.zeros((n, n)), np.zeros((n, n)), np.eye(n)])
        A_blocks.append(A_guard)
        l_list.append(guard_rhs)
        u_list.append(np.full(n, np.inf))

        # 5. Hyperglycemia constraint
        A_hyper = np.hstack([G, np.zeros((n, n)), -np.eye(n), np.zeros((n, n))])
        b_hyper = self.G_max - g0 - self.kappa_hyper * sigma
        A_blocks.append(A_hyper)
        l_list.append(np.full(n, -np.inf))
        u_list.append(b_hyper)

        # 6. Slack nonnegativity
        A_sh = np.hstack([np.zeros((n, n)), -np.eye(n), np.zeros((n, n)), np.zeros((n, n))])
        A_blocks.append(A_sh)
        l_list.append(np.full(n, -np.inf))
        u_list.append(np.zeros(n))

        A_sH = np.hstack([np.zeros((n, n)), np.zeros((n, n)), -np.eye(n), np.zeros((n, n))])
        A_blocks.append(A_sH)
        l_list.append(np.full(n, -np.inf))
        u_list.append(np.zeros(n))

        A_sguard = np.hstack([np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n)), -np.eye(n)])
        A_blocks.append(A_sguard)
        l_list.append(np.full(n, -np.inf))
        u_list.append(np.zeros(n))

        A = np.vstack(A_blocks)
        l = np.concatenate(l_list)
        u = np.concatenate(u_list)

        P_sparse = sp.csc_matrix(np.triu((P + P.T) / 2))
        A_sparse = sp.csc_matrix(A)

        solver = osqp.OSQP()
        solver.setup(P_sparse, q, A_sparse, l, u, verbose=False, warm_start=True)
        result = solver.solve()
        if result.info.status_val not in (1, 2):  # optimal or solved inaccurate
            return np.full(n, self.last_u), np.zeros(n), result.info.status, int(result.info.iter)

        sol = result.x[:n]
        glucose_pred = G @ sol + g0
        return sol, glucose_pred, result.info.status, int(result.info.iter)

    def solve(
        self,
        state_mean: np.ndarray,
        state_cov: np.ndarray,
        meal_schedule: Optional[Dict[int, float]] = None,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        start_time = time.perf_counter()
        meal_schedule = meal_schedule or {}
        F, B = self.dt_model._linearize_dynamics(state_mean, self.last_u)
        if B.ndim == 1:
            B = B.reshape(self.n_states, 1)
        sigma = np.sqrt(np.clip(self._predict_variances(state_cov, F), 1e-6, None))
        G, g0 = self._prediction_matrices(state_mean, F, B, meal_schedule)
        u_seq, glucose_pred, status, iterations = self._solve_qp(G, g0, sigma)
        u_opt = np.clip(u_seq[0], self.u_min, self.u_max)
        self.last_u = u_opt
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_solve_iterations = iterations
        self.last_solve_status = status
        return u_opt, u_seq, glucose_pred, sigma

    def reset(self):
        basal_u_hr = self.dt_model.patient_params.get('basal_rate', 1.0)
        self.basal_u = basal_u_hr / 60.0
        self.last_u = self.basal_u
        self.osqp_solver = None
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "not_run"


class DeterministicMPC:
    """Deterministic MPC solved with the same OSQP backend (no chance constraints)."""

    def __init__(self, digital_twin: StochasticDigitalTwin):
        self.dt_model = digital_twin
        self.N = config.MPC_HORIZON
        self.n_states = digital_twin.n_states
        self.measurement_index = 1
        self.C_row = np.zeros(self.n_states)
        self.C_row[self.measurement_index] = 1.0

        self.w_glucose = config.W_GLUCOSE
        self.w_insulin = config.W_INSULIN
        self.w_delta_u = config.W_DELTA_U

        self.G_target = config.TARGET_GLUCOSE
        self.G_min = config.G_MIN
        self.G_max = config.G_MAX

        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        self.delta_u_max = config.DELTA_U_MAX

        basal_u_hr = digital_twin.patient_params.get('basal_rate', 1.0)
        self.basal_u = basal_u_hr / 60.0
        self.last_u = self.basal_u

        self.osqp_solver = None
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "not_run"

    def _meal_vector(self, meal_cho: float) -> np.ndarray:
        vec = np.zeros(self.n_states)
        if meal_cho <= 0:
            return vec
        grams = meal_cho * 1000.0
        vec[6] += grams * 0.5
        vec[7] += grams * 0.5
        return vec

    def _prediction_matrices(
        self, state_mean: np.ndarray, F: np.ndarray, B: np.ndarray, meal_schedule: Dict[int, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        S = np.zeros((self.N + 1, self.n_states, self.N))
        v = np.zeros((self.N + 1, self.n_states))
        v[0] = state_mean
        for k in range(self.N):
            v[k] = v[k] + self._meal_vector(meal_schedule.get(k, 0.0))
            S[k + 1] = F @ S[k]
            S[k + 1][:, k] += B[:, 0]
            v[k + 1] = F @ v[k]
        coeff = S[1:]
        const = v[1:]
        G = np.zeros((self.N, self.N))
        g0 = np.zeros(self.N)
        for k in range(self.N):
            G[k, :] = self.C_row @ coeff[k]
            g0[k] = self.C_row @ const[k]
        return G, g0

    def _build_difference_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        D = np.zeros((self.N, self.N))
        D[0, 0] = 1.0
        for k in range(1, self.N):
            D[k, k] = 1.0
            D[k, k - 1] = -1.0
        d_ref = np.zeros(self.N)
        d_ref[0] = self.last_u
        return D, d_ref

    def _solve_qp(self, G: np.ndarray, g0: np.ndarray) -> Tuple[np.ndarray, str, int]:
        n = self.N
        D, d_ref = self._build_difference_matrix()
        target_vec = np.full(n, self.G_target)
        basal_vec = np.full(n, self.basal_u)

        H_u = 2 * (
            self.w_glucose * (G.T @ G)
            + self.w_insulin * np.eye(n)
            + self.w_delta_u * (D.T @ D)
        ) + 1e-6 * np.eye(n)
        f_u = (
            2 * self.w_glucose * (G.T @ (g0 - target_vec))
            - 2 * self.w_insulin * basal_vec
            - 2 * self.w_delta_u * (D.T @ d_ref)
        )

        P = sp.csc_matrix(np.triu((H_u + H_u.T) / 2))
        q = f_u

        A_blocks = []
        l_list = []
        u_list = []

        # Bounds
        A_bounds = np.eye(n)
        A_blocks.append(A_bounds)
        l_list.append(np.full(n, self.u_min))
        u_list.append(np.full(n, self.u_max))

        # Rate limits
        A_rate = D
        A_blocks.append(A_rate)
        l_list.append(-self.delta_u_max + d_ref)
        u_list.append(self.delta_u_max + d_ref)

        # Glucose hard constraints: G u + g0 between [G_min, G_max]
        A_glucose = G
        A_blocks.append(A_glucose)
        l_list.append(self.G_min - g0)
        u_list.append(self.G_max - g0)

        A = sp.csc_matrix(np.vstack(A_blocks))
        l = np.concatenate(l_list)
        u = np.concatenate(u_list)

        solver = osqp.OSQP()
        solver.setup(P, q, A, l, u, verbose=False, warm_start=True)
        res = solver.solve()
        if res.info.status_val not in (1, 2):
            return np.full(n, self.last_u), res.info.status, int(res.info.iter)

        return res.x, res.info.status, int(res.info.iter)

    def solve(self, state_mean: np.ndarray, meal_schedule: Optional[Dict[int, float]] = None) -> float:
        start_time = time.perf_counter()
        meal_schedule = meal_schedule or {}
        F, B = self.dt_model._linearize_dynamics(state_mean, self.last_u)
        if B.ndim == 1:
            B = B.reshape(self.n_states, 1)
        G, g0 = self._prediction_matrices(state_mean, F, B, meal_schedule)
        u_seq, status, iterations = self._solve_qp(G, g0)
        u_opt = np.clip(u_seq[0], self.u_min, self.u_max)
        self.last_u = u_opt
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_solve_iterations = iterations
        self.last_solve_status = status
        return u_opt

    def reset(self):
        basal_u_hr = self.dt_model.patient_params.get('basal_rate', 1.0)
        self.basal_u = basal_u_hr / 60.0
        self.last_u = self.basal_u
        self.osqp_solver = None
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "not_run"


class PIDController:
    """Tuned PID baseline."""

    def __init__(self, cohort: str = 'adult'):
        gains = config.PID_GAINS.get(cohort, config.PID_GAINS['adult'])
        self.Kp = gains['Kp']
        self.Ki = gains['Ki']
        self.Kd = gains['Kd']
        self.G_target = config.TARGET_GLUCOSE
        self.u_min = config.U_MIN
        self.u_max = config.U_MAX
        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = config.SAMPLING_TIME
        self.integral_max = 500.0
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "direct_feedback"

    def solve(self, glucose: float) -> float:
        start_time = time.perf_counter()
        error = glucose - self.G_target
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        derivative = (error - self.prev_error) / self.dt

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        if glucose < 85:
            u *= 0.6
        if glucose < 70:
            u = 0.0

        u_saturated = np.clip(u, self.u_min, self.u_max)
        if (u_saturated >= self.u_max and error > 0) or (u_saturated <= self.u_min and error < 0):
            self.integral -= error * self.dt

        self.prev_error = error
        self.last_solve_time_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_solve_iterations = 0
        self.last_solve_status = "direct_feedback"
        return u_saturated

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_solve_time_ms = 0.0
        self.last_solve_iterations = 0
        self.last_solve_status = "direct_feedback"
