"""Custom exceptions that can be raised during optimization. """

class ConvergenceError(Exception):
    """Error to raise when a circuit does not converge during a process, like
    optimization. Initialized with the computed value for epsilon from the
    convergence test. 
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def __str__(self):
        return f'Your circuit did not converge. The computed epsilon was {self.epsilon}.'
