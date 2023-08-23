from abc import abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence
from typing import Optional, Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import comb
from scipy.stats import qmc
import sys

from .utils import create_sampler
from .truncation import trunc_num_heuristic, test_convergence

import SQcircuit as sq

IndexingType: TypeAlias = Union[list[int], np.ndarray]

class Swarm(Protocol):
    """
    The PSO optimiser operates on the abstraction that the position is just 
    a matrix X of vectors. This provides a way to change how that works.
    """
    @abstractproperty
    def position(self) -> np.ndarray:
        """
        Returns a numpy array with shape `(self.swarm_size, self.dimensions)`.
        """
        ...

    @abstractproperty
    def swarm_size(self) -> int:
        ...

    @abstractproperty
    def dimensions(self) -> int:
        ...

    @abstractmethod
    def set_position(self, X) -> None:
        """
        Accepts a numpy array `X` with shape `(self.swarm_size, self.dimensions)`.
        """
        ...

    @abstractmethod
    def eval_position(self, 
                      to_eval: Optional[IndexingType] = None
                      ) -> np.ndarray:
        """
        Returns a numpy array with shape `(self.swarm_size,)`. If `to_eval` is
        not None, only evaluates the particles given by `to_eval`.
        """
        ...


class BasicSwarm(Swarm):
    def __init__(self, swarm_size: int, 
                 dimensions: int, 
                 init_constraints,
                 loss_func,
                 init_strat: str = 'uniform',
                 seed: Optional[int] = None):
        self._swarm_size: int = swarm_size
        self._dimensions: int = dimensions
        self.loss_func = loss_func

        self.rng = np.random.default_rng(seed=seed)
                     
        if len(init_constraints) != self._dimensions:
            raise ValueError('Length of initialisation must be equal to the number of dimensions!')
        else:
            self.init_constraints = init_constraints
            self.lower_bound = np.array([bound[0] for bound in init_constraints])
            self.upper_bound = np.array([bound[1] for bound in init_constraints]) 

        self.init_strat = init_strat
        self._init_pos()

    def _init_pos(self) -> None:
        if self.init_strat == 'uniform':
            self.X = np.zeros((self.swarm_size, self.dimensions))
            self.X = self.rng.uniform(self.lower_bound, self.upper_bound,
                                      size=(self.swarm_size, self.dimensions))
        elif self.init_strat == 'loguniform':
            self.X = np.zeros((self.swarm_size, self.dimensions))
            try:
                log_lower_bound = np.log(self.lower_bound)
                log_upper_bound = np.log(self.upper_bound)
            except RuntimeWarning:
                sys.exit('All bounds must be positive for log-uniform sampling.')
            self.X = np.exp(self.rng.uniform(log_lower_bound, log_upper_bound,
                                             size=(self.swarm_size, self.dimensions)))          
        else:
            raise ValueError(f'{self.init_strat} is not a valid initialisation strategy.')

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def swarm_size(self) -> int:
        return self._swarm_size

    @property
    def position(self) -> np.ndarray:
        return self.X
        
    def set_position(self, X: np.ndarray) -> None:
        self.X = X

    def eval_position(self, 
                      to_eval: Optional[IndexingType] = None
                      ) -> np.ndarray:
        if to_eval is None:
            out = np.zeros(self._swarm_size)
            for i in range(len(out)):
                out[i] = self.loss_func(self.X[i,:])
        else:
            out = np.zeros_like(to_eval)
            for out_idx, particle_idx in enumerate(to_eval):
                out[out_idx] = self.loss_func(self.X[particle_idx,:])
        return out

class CircuitSwarm(Swarm):
    """

    If `conserve_memory` is False, we maintain a population of `num_circuits`
    distinct `Circuit` objects. Otherwise only the parameters are maintained 
    internally, and we initialise a new circuit for each one when necessary.
    """
    def __init__(self, 
                 num_circuits: int, 
                 circuit_code: str,
                 capacitor_range, inductor_range, junction_range,
                 num_eigenvalues: int = 10, total_trunc_num: int = 140,
                 sampling_method: str = 'loguniform', is_log: bool = False,
                 parallel: bool = False, conserve_memory: bool = False):
                     
        self._num_circuits = num_circuits
        self._circuit_code = circuit_code
        self._num_inductive_elements = len(circuit_code)
        self._num_elements = self.num_elems(circuit_code)

        self.capacitor_range = capacitor_range
        self.inductor_range = inductor_range
        self.junction_range = junction_range

        self.num_eigenvalues: int = num_eigenvalues
        self.total_trunc_num: int = total_trunc_num

        self.conserve_memory: bool = conserve_memory
        self._sampling_method: str= sampling_method
        self._sample_circuits()
        self.history = []

        self.is_log: bool = is_log

    def _get_actual_bounds(self):
        bounds = []
        for elem in self.ordered_elements(self.model_circuit):
            if isinstance(elem, sq.elements.Junction):
                bounds.append(self.junction_range)
            elif isinstance(elem, sq.elements.Inductor):
                bounds.append(self.inductor_range)
            elif isinstance(elem, sq.elements.Capacitor):
                bounds.append(self.capacitor_range)
            else:
                bounds.append(None)
        return bounds

    def get_bounds(self):
        bounds = self._get_actual_bounds()

        if self.is_log:
            return [np.log10(b) for b in bounds]
        return bounds

    @staticmethod
    def voroni_sampling(k: int, n: int) -> np.ndarray:
        """
        Sample `k` points in the `n`-dimenisional unit hypercube with an 
        approximation of a central Voroni tessellation. 
        """
        alpha_1 = 0.5 #np.random.uniform(0, 1) #0
        alpha_2 = 1 - alpha_1
        beta_1 = 0.5 #np.random.uniform(0, 1) #0
        beta_2 = 1 - beta_1
    
        num_iter = 10
        q = 50_000
    
        pts = np.random.uniform(0, 1, size=(k, n))
        c = np.full(k, 1)
        for iter in range(num_iter):
            samples = np.random.uniform(0, 1, size=(q, n))
            distance_to_pts = np.argmin(np.linalg.norm(np.stack([samples for _ in range(k)], axis=1) - pts, axis=2), axis=1)
            for pt_idx in range(k):
                closest_samples = (distance_to_pts == pt_idx)
                if not np.any(closest_samples):
                    break
                mean_pt = np.mean(samples[closest_samples, :], axis=0)
                pts[pt_idx,:] = ((alpha_1 * c[pt_idx] + beta_1) * pts[pt_idx,:] \
                                 + (alpha_2 * c[pt_idx] + beta_2) * mean_pt)/(c[pt_idx] + 1)
                c[pt_idx] += 1
        return pts

    @staticmethod
    def sobol_sampling(k: int, n: int) -> np.ndarray:
        """
        Sample `k` points in the `n`-dimensional unit hypercube using a Sobol
        sequence.
        """
        nec_bits = int(np.ceil(np.log2(k)))
        if nec_bits > 62:
            raise ValueError('Too many particles requested.')
            
        if nec_bits > 28: ## default number of bits is 30
            sampler = qmc.Sobol(d=n, bits=(nec_bits + 2))
        else:
            sampler = qmc.Sobol(d=n)
        return sampler.random(k)

    def _sample_circuits(self) -> None:
        # Retain circuit/parameters internally
        if self.conserve_memory:
            self.circuit_params = np.zeros((self._num_circuits, self._num_elements))
        else:
            self.circuits = []
        self.circuit_metadata = []
        
        # Construct circuits
        for i in range(self._num_circuits):
            sampler = create_sampler(self._num_inductive_elements, self.capacitor_range, 
                                     self.inductor_range, self.junction_range)
            circuit = sampler.sample_circuit_code(self._circuit_code)

            if i==0:
                self.model_circuit = circuit

            # Choose baseline truncation numbers based on total_trunc_num
            trunc_nums = circuit.truncate_circuit(self.total_trunc_num)

            if self.conserve_memory:
                self.circuit_params[i,:] = self.get_values(circuit)
            else:
                self.circuits.append(circuit)
            self.circuit_metadata.append((trunc_nums, None))

        # Sample values in unit hypercube
        if self._sampling_method == 'loguniform':
            return
        elif self._sampling_method == 'cvt':
            new_vals = self.voroni_sampling(self._num_circuits, self._num_elements)
        elif self._sampling_method == 'sobol':
            new_vals = self.sobol_sampling(self._num_circuits, self._num_elements)
        else:
            raise ValueError('Invalid sampling method.')

        # Scale to be log-distributed
        bounds = self._get_actual_bounds()
        log_bounds = [np.log(b) for b in bounds]
        b_range = np.array([(b[1] - b[0]) for b in log_bounds])
        b_offset = np.array([b[0] for b in log_bounds])
        new_vals = np.exp(new_vals * b_range + b_offset)

        # Update circuits
        self._set_actual_position(new_vals)

    @property
    def dimensions(self) -> int:
        return self._num_elements

    @property
    def swarm_size(self) -> int:
        return self._num_circuits

    @staticmethod
    def num_elems(circuit_code: str) -> int:
        """
        Return number of elements in loop circuit with all-to-all capacitive
        coupling.
        """
        inductive_elems = len(circuit_code)
        capactive_elems = comb(inductive_elems, 2)
        return int(inductive_elems + capactive_elems)

    @staticmethod
    def ordered_elements(circuit: sq.Circuit) -> Iterator[sq.Element]:
        for edge in circuit.elements.values():
            for elem in edge:
                yield elem

    def get_values(self, cr: sq.Circuit) -> np.ndarray:
        vals = []
        for elem in self.ordered_elements(cr):
            vals.append(elem.get_value())
        return np.array(vals)

    @property
    def position(self) -> np.ndarray:
        if self.conserve_memory:
            X = self.circuit_params.copy()
        else:
            X = np.zeros((self.swarm_size, self.dimensions))
            for i, cr in enumerate(self.circuits):
                for j, elem in enumerate(self.ordered_elements(cr)):
                    X[i,j] = elem.get_value()

        if self.is_log:
            return np.log10(X)
        return X

    def set_circuit_params(self, cr: sq.Circuit, params) -> None:
        for i, elem in enumerate(self.ordered_elements(cr)):
            ## UPDATE IF JUNCTION `set_value` FIXED
            elem._value = params[i]

    def _set_actual_position(self, X: np.ndarray) -> None:
        if self.conserve_memory:
            self.circuit_params = X.copy()
        else:
            for i, cr in enumerate(self.circuits):
                self.set_circuit_params(cr, X[i,:])
                cr.update()

    def set_position(self, X: np.ndarray) -> None:
        if self.is_log:
            self._set_actual_position(np.power(10, X))
        else:
            self._set_actual_position(X)

    def _diag_circuit(self, cr: sq.Circuit, trunc_nums: int) -> tuple[int, bool]:
        # TODO: check this is all correct re: convergence checking
        # update with verify_convergence from truncation.py, if desired
        cr.diag(self.num_eigenvalues)
        if len(cr.m) == 1:
            is_converged = cr.test_convergence(trunc_nums)
        elif len(cr.m) == 2:
            trunc_nums = trunc_num_heuristic(cr,
                                             K=self.total_trunc_num,
                                             eig_vec_idx=1,
                                             axes=None)
            cr.set_trunc_nums(trunc_nums)
            cr.diag(self.num_eigenvalues)

            is_converged, _, _ = test_convergence(cr, eig_vec_idx=1)

        return trunc_nums, is_converged

    def _calc_loss(self, cr: sq.Circuit):
        total_loss, loss_values = calculate_loss(cr)
        metric_values = calculate_metrics(cr) + (total_loss, )
        return total_loss, loss_values, metric_values
    
    def eval_position(self, 
                      to_eval: Optional[IndexingType] = None
                      ) -> np.ndarray:
        tot_losses = np.zeros(self._num_circuits)
        all_metrics = []
        all_losses = []
        if to_eval is None:
            to_eval = np.full(self._num_circuits, True)
        for idx in range(self._num_circuits):
            if to_eval[idx]:
                # Calculate if necessary
                if self.conserve_memory:
                    self.set_circuit_params(self.model_circuit, 
                                            self.circuit_params[idx,:])
                    self.model_circuit.update()
                    cr = self.model_circuit
                else:
                    cr = self.circuits[idx]
    
                trunc_nums, is_converged = self._diag_circuit(cr,
                                            self.circuit_metadata[idx][0])
                self.circuit_metadata[idx] = [trunc_nums, is_converged]
    
                if is_converged:
                    total_loss, loss_values, metric_values = self._calc_loss(cr)
                    all_metrics.append(metric_values)
                    all_losses.append(loss_values)
                else:
                    total_loss = np.inf
                    metric_values = None
                    loss_values = None
            else:
                # Otherwise pull from history
                metric_values = self.history[-1][idx][0]
                loss_values = self.history[-1][1]
                total_loss = metric_values[-1] if metric_values is not None else np.inf
            # record
            tot_losses[idx] = total_loss
            all_metrics.append(metric_values)
            all_losses.append(loss_values)
        self.history.append((all_metrics, all_losses))
        return tot_losses[to_eval]
    
class Optimiser(Protocol):
    @abstractmethod
    def is_converged(self) -> bool:
        ...

    @abstractmethod
    def optimise(self) -> None:
        ...

class PSO(Optimiser):
    """
    Variants are
        - Inertial weighting
        - Constriction factor

    Topologies are
        - gbest
        - lbest
        - von Neumann

    Not yet implemented
        - lbest-closest

    For the moment constraints are required for possible parameters.
    """
    def __init__(self, 
                 constraints: Sequence[tuple[float, float]], 
                 swarm: Swarm,
                 v_start_delta: float = 0,
                 variant: str = 'constrict', 
                 params: dict[str, float] = {'c1': 2.05, 'c2': 2.05, 'kappa': 1},
                 max_iter: int = 100, 
                 seed: Optional[int] = None,
                 bounds_handler: str = 'clamp', 
                 velocity_delta: Optional[float] = None,
                 nbhd_topology: str='gbest', 
                 nbhd_params: Optional[dict[str, float]]= None,
                 verbose: bool = True, 
                 cache: bool = False):

        self.swarm = swarm
        self.dimensions = self.swarm.dimensions
        self.swarm_size = self.swarm.swarm_size

        self.nbhd_topology = nbhd_topology
        if self.nbhd_topology == 'lbest':
            assert nbhd_params is not None
            self.left_nbhs = int(nbhd_params['n'] // 2)
            self.right_nbhs = int(nbhd_params['n'] - self.left_nbhs)
            self.nbhd_indices = self.nearest_index_nbhs(self.swarm_size,
                                                        self.left_nbhs,
                                                        self.right_nbhs)
        elif self.nbhd_topology == 'von-neumann':
            assert nbhd_params is not None
            self.nbhd_indices = self.von_neumann_nbhs(self.swarm_size)
            
        self.variant = variant
        self.params = params
        if self.variant == 'constrict':
            self.params['phi'] = self.params['c1'] + self.params['c2']
            self.params['chi'] = (2 * self.params['kappa']) \
                                 / np.abs(2 - self.params['phi'] - np.sqrt(self.params['phi'] * (self.params['phi'] - 4)))

        if len(constraints) != self.dimensions:
            raise ValueError('Length of constraints must be equal to the number of dimensions!')
        else:
            self.constraints = constraints
            self.lower_bound = np.array([bound[0] for bound in constraints])
            self.upper_bound = np.array([bound[1] for bound in constraints])

        self.velocity_delta = velocity_delta
        if self.velocity_delta is not None:
            self.max_velocity = self.velocity_delta * (self.upper_bound - self.lower_bound)

        self.bounds_handler = bounds_handler

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)   

        self.X = self.swarm.position                      
        self.V = np.zeros((self.swarm_size, self.dimensions))
        if v_start_delta > 0:
            tmp_V = self.sample_sphere(self.swarm_size, self.dimensions)
            tmp_V *= v_start_delta * (self.upper_bound - self.lower_bound)
            self.V = tmp_V
        self.pbest: Optional[np.ndarray] = None
        self.pbest_cost: Optional[np.ndarray] = None

        self.history: list[dict[str, np.ndarray]] = []
        self.max_iter = max_iter
        self.iters_completed = 0

        self.verbose = verbose

        # depending on neighbourhood topology, particles may not move for
        # several iterations (which is its own problem), in which case it 
        # is wasteful to re-evaluate the function.
        self.cache = cache
        if self.cache:
            self.old_X: Optional[np.ndarray] = None
            self.pcurr_cost = np.full(self.swarm_size, np.inf)

    def sample_sphere(self, n: int, d: int) -> np.ndarray:
        """
        Samples `n` points uniformly over the surface of `d`-dimensional
        unit sphere. Returns a `(n, d)` Numpy array.
        """
        points = self.rng.normal(size=(n, d))
        points = points / np.linalg.norm(points, axis=1)[:,np.newaxis]
        return points

    def is_converged(self) -> bool:
        """
        Returns whether the swarm has converged. Currently not implemented.
        """
        return False

    def _record_pos(self) -> None:
        """
        Record the position and current loss values into `self.history`.
        """
        # Only call after evaluating best positions
        assert self.pbest is not None
        assert self.pbest_cost is not None

        # TODO: Write to file intermittently?
        self.history.append({'X': self.X.copy(), 
                             'V': self.V.copy(), 
                             'pcurr_cost': self.pcurr_cost.copy(),
                             'pbest': self.pbest.copy(), 
                             'pbest_cost': self.pbest_cost.copy(), 
                             'sbest': self.sbest.copy(), 
                             'sbest_cost': self.sbest_cost.copy()})

    @staticmethod
    def nearest_index_nbhs(arr_length: int, 
                           left_nbh_count: int, 
                           right_nbh_count: int
                           ) -> np.ndarray:
        """
        Returns an `(arr_length, left_nbh_count + right_nbh_count + 1)` array
        where the ith row has the indices 
            `left_nbh_count`, `left_nbh_count` + 1, ..., i, ..., 
            `right_nbh_count` - 1, `right_nbh_count`
        (wrapping on `arr_length`). 
        """
        out = np.tile(np.arange(-left_nbh_count,right_nbh_count+1),
                      (arr_length, 1))
        out += np.arange(arr_length)[:,np.newaxis]
        return np.mod(out, arr_length)

    @staticmethod
    def von_neumann_nbhs(ns: int) -> np.ndarray:
        """
        Returns a `(ns, 5)` numpy array where the ith row gives the adjacent
        edges to the ith vertex in an approximation of a "von Neummman" graph
        on `ns` vertices, where we consider a vertex to be self-adjacent.

        To approximate a von Neumann graph, arrange the `ns` numbers
        in a ceil(sqrt(ns)) by ceil(sqrt(ns)) grid and connecting to N/E/S/W
        neighbours, wrapping where necessary.
        """
        # Initialise output array
        out = np.zeros((ns, 5), dtype='int')
    
        # Calculate num_columns and num_rows to get the smallest grid
        # that has `ceil(sqrt(ns))`` columns and at least `ns` spaces.
        num_columns = np.ceil(np.sqrt(ns)).astype(int)
        num_rows = np.ceil(ns/num_columns).astype(int)
        extra_spaces = (num_columns * num_rows) - ns
    
        # If `row_lengths* column_spaces > ns`, there will be extra spaces
        # in the grid. Assume they're all at the end (in row-major order)
        # and calculate the actual row and column lengths for a ragged grid
        # when these are omitted
        row_lengths = np.full(num_rows, num_columns)
        row_lengths[-1] = ns - ((num_rows - 1) * num_columns)
        column_lengths = np.full(num_columns, num_rows)
        if extra_spaces > 0:
            column_lengths[-extra_spaces:] = num_rows - 1
        
        # Iterate through all spaces in the ragged grid and find the 
        # N/E/S/W neighbours, wrapping as necessary
        for row_idx in range(num_rows):
            for column_idx in range(num_columns):
                # calculate the current position
                # this is always accurate even though the grid is 'ragged'
                # since we fill *and* iterate in row-major order
                n = row_idx * num_columns + column_idx
                # self
                out[n, 0] = n
                # E
                out[n, 1] = row_idx * num_columns \
                            + ((column_idx + 1) % row_lengths[row_idx]) 
                # W
                out[n, 2] = row_idx * num_columns \
                            + ((column_idx - 1) % row_lengths[row_idx])
                # N
                out[n, 3] = ((row_idx + 1) % column_lengths[column_idx]) * num_columns \
                            + column_idx
                # S
                out[n, 4] = ((row_idx - 1) % column_lengths[column_idx]) * num_columns \
                            + column_idx
        
                if n == ns - 1: return out
        raise ValueError(f"Something's gone wrong with intput ns={ns}")

    def get_best_nbh_idx(self) -> np.ndarray:
        """
        Returns an array of the neighbour with the lowest `pbest_cost` for each 
        particle in the swarm, where the set of neighbours are determined by
        `self.nbhd_indices`.
        """
        assert self.pbest_cost is not None

        return self.nbhd_indices[np.arange(self.swarm_size),
                                 np.argmin(self.pbest_cost[self.nbhd_indices],
                                            axis=1)]
    
    def _update_best(self) -> None:
        """
        Updates ther personal best (`self.pbest`) and social best 
        (`self.sbest`) positions. 
        
        The personal best position is always best position seen by the
        particle. The social best position depends on the neighbourhood
        topology (`self.nbhd_topology`).
        """
        if self.pbest_cost is not None:
            assert self.pbest is not None # should always match self.pbest_cost
            update_mask = self.pcurr_cost <= self.pbest_cost
            self.pbest[update_mask, :] = self.X[update_mask, :]
            self.pbest_cost[update_mask] = self.pcurr_cost[update_mask]
        else:
            self.pbest = self.X.copy()
            self.pbest_cost = self.pcurr_cost.copy()

        if self.nbhd_topology == 'gbest':
            self.sbest = self.pbest[np.argmin(self.pbest_cost), :]
            self.sbest_cost = np.min(self.pbest_cost)
        elif self.nbhd_topology == 'lbest' or self.nbhd_topology == 'von-neumann':
            best_nbh_idx = self.get_best_nbh_idx()
            self.sbest = self.pbest[best_nbh_idx,:]
            self.sbest_cost = self.pbest_cost[best_nbh_idx]           

    def _update_velocity(self) -> None:
        """
        Updates the velocity of the swarm based on the velocity update rule
        given in `self.variant`. Does any velocity clamping determined by
        `self.velocity_delta`.
        """
        # Must calculate `pbest` first
        assert self.pbest is not None
        assert self.pbest_cost is not None

        # Update (putative velocity)
        if self.variant == 'constrict':
            r1 = self.rng.uniform(size=(self.swarm_size, self.dimensions))
            r2 = self.rng.uniform(size=(self.swarm_size, self.dimensions))
            V_temp = self.params['chi'] * (self.V \
                                           + self.params['c1'] * r1 * (self.pbest - self.X) \
                                           + self.params['c2'] * r2 * (self.sbest - self.X))
        if self.variant == 'weight' or self.variant == 'weight-decrease':
            r1 = self.rng.uniform(size=(self.swarm_size, self.dimensions))
            r2 = self.rng.uniform(size=(self.swarm_size, self.dimensions))
            V_temp = self.params['w'] * self.V \
                        + self.params['c1'] * r1 * (self.pbest - self.X) \
                        + self.params['c2'] * r2 * (self.sbest - self.X)
            if self.variant == 'weight-decrease':
                self.params['w'] -= self.params['w-decrease']
            
        # Clamp velocity, if desired
        if self.velocity_delta is not None:
            V_temp = np.where(V_temp > self.max_velocity, 
                              self.max_velocity * np.sign(V_temp),
                              V_temp)

        self.V = V_temp
    
    def _update_position(self) -> None:
        """
        Updates the position in three steps
            1. Adds `self.V` to `self.X`;
            2. Does any desired boundary handling;
            3. Sets the swarm position to `self.X`.
        """
        X_temp = self.X + self.V
        too_high = X_temp > self.upper_bound
        too_low = X_temp < self.lower_bound
        
        if np.any(too_high) or np.any(too_low):
            if self.verbose:
                print('Had to handle bounds')
            if self.bounds_handler == 'clamp':
                X_temp = np.where(too_low, self.lower_bound, X_temp)
                X_temp = np.where(too_high, self.upper_bound, X_temp)
            if self.bounds_handler == 'random':
                # A little wasteful to generate all of them, but there it is
                random_pos = self.rng.uniform(self.lower_bound, self.upper_bound,
                                          size=(self.swarm_size, self.dimensions))
                X_temp = np.where(too_low | too_high, random_pos, X_temp)
            if self.bounds_handler == 'reflect':
                X_temp = np.where(too_low, 2 * self.lower_bound - X_temp, X_temp)
                X_temp = np.where(too_high, 2 * self.upper_bound - X_temp, X_temp)
            
        self.swarm.set_position(X_temp)
        self.X = X_temp
            
    def _eval_current_pos(self) -> None:
        """
        Evaluates the current position of the swarm and saves it to 
        `self.pcurr_cost`. 
        """
        if self.cache:
            moved = ~np.all(self.X == self.old_X, axis=1)
            self.pcurr_cost[moved] = self.swarm.eval_position(to_eval=moved)
            self.old_X = self.X
        else:
            self.pcurr_cost = self.swarm.eval_position()
    
    def optimise(self, num_iters: Optional[int] = None) -> None:
        """
        Runs particle swarm optimization for a maximum of `num_iters` more 
        iterations, if provided, or `max_iter`, if not. Breaks if
        `self.is_converged()` returns True.
        """
        if num_iters is None:
            num_iters = self.max_iter
        for iter in range(num_iters):
            ## Evaluate current position
            self._eval_current_pos()

            ## Update swarm statistics
            self._update_best()

            ## Save data
            self._record_pos()

            ## Check convergence criteria
            if self.is_converged(): 
                break

            ## Move
            self._update_velocity()
            self._update_position()

            self.iters_completed += 1
            if self.verbose:
                print(f'Finished iteration {iter}/{num_iters}.')
