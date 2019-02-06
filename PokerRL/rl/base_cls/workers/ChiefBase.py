# Copyright (c) 2019 Eric Steinberger


from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class ChiefBase(WorkerBase):
    """
    The Chief is the worker logs are sent to (although they are presented on the Driver worker). It is also responsible
    for tracking what iteration is currently running and tunneling up-to-date strategies between workers upon
    request. Furthermore, the Chief exports EvalAgents and logs to permanent storage.
    """

    def __init__(self, t_prof):
        """
        Args:
            t_prof (TrainingProfile)
        """
        super().__init__(t_prof=t_prof)
        self._experiment_names = {}
        self._log_buf = _LogBuffer()

    def pull_current_eval_strategy(self, last_iteration_receiver_has):
        """
        This function is used to update EvalAgents with the current NN weights / strategy / anything needed.
        The specific outputs of this function are completely dynamic, as they should be tunneled between
        algo-specific components.

        Args:
            last_iteration_receiver_has (int)

        Returns:
            Tuple(Weights, any_other_info)
        """
        raise NotImplementedError

    def export_agent(self, step):
        """
        Wraps the current strategy of the agent in an EvalAgent instance and exports that.
        """
        raise NotImplementedError

    # __________________________________________________ Logging API ___________________________________________________
    def create_experiment(self, name):
        """
        Registers a new experiment in the LogBuffer, which can later be seen and pulled into TensorBoard.

        Args:
            name (str): Name of the new experiment

        Returns:
            name (str): Name of the new experiment
        """
        return self._log_buf.create_experiment(name)

    def add_scalar(self, exp_name, graph_name, step, value):
        """
        Adds one datapoint to a 2D graph of an experiment in the LogBuffer.

        Args:
            exp_name (str): Name of the new experiment
            graph_name (str): Name of the graph into which to plot
            step (int): Timestep (x-axis) of the datapoint
            value (float): Value to plot at timestep ""step"".
        """
        self._log_buf.add_scalar(exp_name=exp_name, graph_name=graph_name, step=step, value=value)

    def get_new_values(self):
        return self._log_buf.get_new_values()


class _LogBuffer:
    """
    This class STORES logs. It does not write them to TensorBoard; for that use CrayonWrapper.
    """

    def __init__(self):
        self._experiments = {}
        self._new_values = {}

    def clear(self):
        self._experiments = {}

    def create_experiment(self, name):
        if name not in self._experiments.keys():
            self._experiments[name] = {}
        return name  # returning name for convenience when creating experiments in a list comp

    def add_scalar(self, exp_name, graph_name, step, value):
        if exp_name not in self._experiments.keys():
            raise AttributeError("Should create experiment before adding to it")

        if graph_name not in self._experiments[exp_name].keys():
            self._experiments[exp_name][graph_name] = []

        self._experiments[exp_name][graph_name].append([step, value])

        if exp_name not in self._new_values.keys():
            self._new_values[exp_name] = {}
        if graph_name not in self._new_values[exp_name].keys():
            self._new_values[exp_name][graph_name] = []

        self._new_values[exp_name][graph_name].append([step, value])

    def get_new_values(self):
        new_v = self._new_values
        self._new_values = {}
        experiment_names = list(self._experiments.keys())
        return new_v, experiment_names
