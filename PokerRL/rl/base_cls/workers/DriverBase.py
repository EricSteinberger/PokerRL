# Copyright (c) 2019 Eric Steinberger


import shutil
from os.path import join as ospj

from PokerRL._.CrayonWrapper import CrayonWrapper
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase
from PokerRL.util import file_util


class DriverBase(WorkerBase):
    """
    The Driver runs the HighLevelAlgo and creates all workers. If running distributed, the Driver is the worker node
    that launches the computation and distributes tasks between workers.
    """

    def __init__(self, t_prof, eval_methods, chief_cls, eval_agent_cls, n_iterations=None, iteration_to_import=None,
                 name_to_import=None):
        """
        Args:
            t_prof (TrainingProfile)
            eval_methods (dict):                dict of {evaluator1_name: frequency, ...} Currently supported evaluators
                                                are "br", "rlbr", and "lbr"
            chief_cls (ChiefBase subclass):     class, not instance
            n_iterations (int)                  number of iterations to run. If None, runs forever
            iteration_to_import (int):               step/iteration to import
            name_to_import (str):               name of the run to import
        """
        super().__init__(t_prof=t_prof)

        if self._t_prof.CLUSTER:
            self._ray.init_cluster(redis_address=t_prof.redis_head_adr)
        else:
            self._ray.init_local()

        file_util.do_pickle(obj=t_prof, file_name=t_prof.name, path=t_prof.path_trainingprofiles)
        self.n_iterations = n_iterations

        self._step_to_import = iteration_to_import
        self._name_to_import = name_to_import

        if self._t_prof.DISTRIBUTED:
            from PokerRL.eval.lbr.DistLBRMaster import DistLBRMaster as LBRMaster
            from PokerRL.eval.rl_br.DistRLBRMaster import DistRLBRMaster as RLBRMaster
            from PokerRL.eval.rl_br.workers.ps.Dist_RLBR_ParameterServer import \
                Dist_RLBR_ParameterServer as RLBRParameterServer
            from PokerRL.eval.rl_br.workers.la.Dist_RLBR_LearnerActor import Dist_RLBR_LearnerActor as RLBRLearnerActor
            from PokerRL.eval.lbr.DistLBRWorker import DistLBRWorker as LBRWorker
            from PokerRL.eval.br.DistBRMaster import DistBRMaster as BRMaster
            from PokerRL.eval.head_to_head.DistHead2HeadMaster import DistHead2HeadMaster as Head2HeadMaster

        else:
            from PokerRL.eval.lbr.LocalLBRMaster import LocalLBRMaster as LBRMaster
            from PokerRL.eval.rl_br.LocalRLBRMaster import LocalRLBRMaster as RLBRMaster
            from PokerRL.eval.rl_br.workers.ps.Local_RLBR_ParameterServer import \
                Local_RLBR_ParameterServer as RLBRParameterServer
            from PokerRL.eval.rl_br.workers.la.Local_RLBR_LearnerActor import \
                Local_RLBR_LearnerActor as RLBRLearnerActor
            from PokerRL.eval.lbr.LocalLBRWorker import LocalLBRWorker as LBRWorker
            from PokerRL.eval.br.LocalBRMaster import LocalBRMaster as BRMaster
            from PokerRL.eval.head_to_head.LocalHead2HeadMaster import LocalHead2HeadMaster as Head2HeadMaster

        # safety measure to avoid overwriting logs when reloading
        if name_to_import is not None and iteration_to_import is not None and name_to_import == t_prof.name:
            t_prof.name += "_"

        print("Creating Chief...")
        self.chief_handle = self._ray.create_worker(chief_cls,
                                                    t_prof)

        self.eval_masters = {}
        if "br" in list(eval_methods.keys()):
            print("Creating BR Evaluator...")
            self.eval_masters["br"] = (self._ray.create_worker(BRMaster,
                                                               t_prof,
                                                               self.chief_handle,
                                                               eval_agent_cls),
                                       eval_methods["br"]  # freq
                                       )

        if "h2h" in list(eval_methods.keys()):
            print("Creating Head-to-Head Mode Evaluator...")
            self.eval_masters["h2h"] = (self._ray.create_worker(Head2HeadMaster,
                                                                t_prof,
                                                                self.chief_handle,
                                                                eval_agent_cls),
                                        eval_methods["h2h"]  # freq
                                        )

        if "lbr" in list(eval_methods.keys()):
            print("Creating LBR Evaluator...")
            self._lbr_workers = [
                self._ray.create_worker(LBRWorker,
                                        t_prof,
                                        self.chief_handle,
                                        eval_agent_cls)
                for _ in range(self._t_prof.module_args["lbr"].n_workers)
            ]

            self.eval_masters["lbr"] = (self._ray.create_worker(LBRMaster,
                                                                t_prof,
                                                                self.chief_handle),
                                        eval_methods["lbr"]  # freq
                                        )
            self._ray.wait([
                self._ray.remote(self.eval_masters["lbr"][0].set_worker_handles,
                                 *self._lbr_workers)
            ])

        if "rlbr" in list(eval_methods.keys()):
            print("Creating RL-BR Evaluator...")
            self._rlbr_ps = self._ray.create_worker(RLBRParameterServer,
                                                    t_prof,
                                                    self.chief_handle,
                                                    )
            self._rlbr_las = [
                self._ray.create_worker(RLBRLearnerActor,
                                        t_prof,
                                        self.chief_handle,
                                        eval_agent_cls
                                        )
                for _ in range(self._t_prof.module_args["rlbr"].n_las)
            ]

            self.eval_masters["rlbr"] = (self._ray.create_worker(RLBRMaster,
                                                                 t_prof,
                                                                 self.chief_handle,
                                                                 eval_agent_cls),
                                         eval_methods["rlbr"]  # freq
                                         )

            self._ray.wait([
                self._ray.remote(self.eval_masters["rlbr"][0].set_learner_actors,
                                 *self._rlbr_las),
            ])
            self._ray.wait([
                self._ray.remote(self.eval_masters["rlbr"][0].set_param_server,
                                 self._rlbr_ps),
            ])

        self.crayon = CrayonWrapper(name=t_prof.name, chief_handle=self.chief_handle,
                                    path_log_storage=self._t_prof.path_log_storage,
                                    crayon_server_address=t_prof.local_crayon_server_docker_address,
                                    runs_distributed=t_prof.DISTRIBUTED,
                                    runs_cluster=t_prof.CLUSTER,
                                    )

    def _maybe_load_checkpoint_init(self):
        if self._step_to_import is None:
            self._cfr_iter = 0
        else:
            assert self._step_to_import is not None
            print("Loading checkpoint ", self._step_to_import)
            self._cfr_iter = self._step_to_import + 1
            self.load_checkpoint(step=self._step_to_import, name_to_load=self._name_to_import)
            print("Loaded from iter: ", self._step_to_import)

    def _delete_past_checkpoints(self, steps_not_to_delete):
        _dir = ospj(self._t_prof.path_checkpoint, self._t_prof.name)
        all_dir_names = file_util.get_all_dirs_in_dir(_dir)
        dir_names_to_delete = [e for e in all_dir_names if e not in [str(s) for s in steps_not_to_delete]]

        for step in dir_names_to_delete:
            shutil.rmtree(ospj(_dir, step))

    def _delete_past_log_files(self, steps_not_to_delete):
        _dir = ospj(self._t_prof.path_log_storage, self._t_prof.name)
        all_dir_names = file_util.get_all_dirs_in_dir(_dir)
        dir_names_to_delete = [e for e in all_dir_names if e not in [str(s) for s in steps_not_to_delete]]

        for step in dir_names_to_delete:
            shutil.rmtree(ospj(_dir, step))

    def evaluate(self):
        """
        puts whole network on wait while the parameters are synced, but evaluates while training the next iteration(s)
        """
        evaluators_to_run = []
        for kind in list(self.eval_masters.keys()):
            ev = self.eval_masters[kind][0]
            freq = self.eval_masters[kind][1]
            if self._cfr_iter % freq == 0:
                evaluators_to_run.append(ev)
                print("Evaluating vs.", kind.upper())
                self._ray.wait([
                    self._ray.remote(ev.update_weights)
                ])

        for ev in evaluators_to_run:
            self._ray.remote(ev.evaluate,
                             self._cfr_iter)

    def save_logs(self):
        """
        Pulls all new logs from the LogBuffer and adds them to crayon (Tensorboard wrapper). It also saves them to disk.
        """
        self.crayon.update_from_log_buffer()
        self.crayon.export_all(iter_nr=self._cfr_iter)

        # Delete past version
        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.log_export_freq + 1:
            s.append(self._cfr_iter - self._t_prof.log_export_freq)

        self._delete_past_log_files(steps_not_to_delete=s)

    def periodically_export_eval_agent(self):
        if self._cfr_iter % self._t_prof.eval_agent_export_freq == 0:
            self.export_eval_agent()

    def periodically_checkpoint(self):
        if self._cfr_iter % self._t_prof.checkpoint_freq == 0:
            print("Saving Checkpoint")
            self.checkpoint(curr_step=self._cfr_iter)

    def export_eval_agent(self):
        print("Exporting agent")
        self._ray.wait([
            self._ray.remote(self.chief_handle.export_agent,
                             self._cfr_iter)
        ])

    # ____________________________________________________ Override ____________________________________________________
    def run(self):
        """
        Calling this function should start up your algorithm.
        """
        raise NotImplementedError

    def checkpoint(self, curr_step):
        """
        Calling this function should trigger "".checkpoint(curr_step=curr_step)"" on all workers in the network.
        """
        raise NotImplementedError

    def load_checkpoint(self, name_to_load, step):
        """
        Calling this function should trigger all workers to load ""name_to_load"" at iteration ""step"" by calling
        worker.load_checkpoint(name_to_load=name_to_load, step=step)
        """
        raise NotImplementedError
