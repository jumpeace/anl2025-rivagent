"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
import itertools
import random
from copy import copy

import numpy as np

from negmas.outcomes import Outcome

#be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
#from helpers.helperfunctions import set_id_dict, ...

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOController, SAOState
from negmas import (
    DiscreteCartesianOutcomeSpace,
    ExtendedOutcome,
    ResponseType, CategoricalIssue,
)

class UtilArea:
    def __init__(self, min_u, max_u, all_bids):
        self.min_u = min_u
        self.u_size = max_u - min_u
        self.max_u = max_u

        self.all_bids = all_bids
        self.n_bids = len(all_bids)
    
        self.min_ru = None
        self.max_ru = None

class UtilModel:
    def __init__(self, ufun, all_bids):
        self.all_bids = all_bids
        self.n_bids = len(all_bids)
        self._bid2u = {str(bid):ufun(bid) for bid in self.all_bids}

        n_area_splits = 5
        area_size = 1.0 / n_area_splits

        bids_tmp = copy(all_bids)
        self._areas = []
        for i in range(n_area_splits-1,-1,-1):
            min_u = area_size * i

            bids_in_area = [bid for bid in bids_tmp if self.get(bid) >= min_u]
            for bid in bids_in_area:
                bids_tmp.remove(bid)

            if len(bids_in_area) == 0:
                continue

            if self.n_areas == 0:
                max_u = None
                for bid in bids_in_area:
                    u = self.get(bid)
                    if (max_u is None) or u > max_u:
                        max_u = u
            else:
                max_u = area_size * (i+1)
            
            ua = UtilArea(min_u, max_u, bids_in_area)
            self._areas.insert(0, ua)
        
        total_u_size = sum([area.u_size for area in self._areas], 0)
        
        self._areas[0].min_ru = 0.0
        self._areas[self.n_areas-1].max_ru = 1.0
        for i in range(self.n_areas-1):
            r = self._areas[i].min_ru + (self._areas[i].u_size / total_u_size)
            self._areas[i].max_ru = r
            self._areas[i+1].min_ru = r
    
    def get(self, bid):
        return self._bid2u[str(bid)]
    
    def get_by_relative(self, relative):
        for i in range(self.n_areas-1,-1,-1):
            min_ru = self._areas[i].min_ru
            if relative >= min_ru:
                r_in_area = (relative - min_ru) / (self._areas[i].max_ru - min_ru)
                return self._areas[i].u_size * r_in_area + self._areas[i].min_u
    
    @property
    def n_areas(self):
        return len(self._areas)
    
    def get_area(self, i):
        return self._areas[i]

class OpponentModel:
    def __init__(self, all_bids):
        self.all_bids = all_bids
        self.n_bids = len(all_bids)
    
    def calc_concession(self):
        return 1.0

class Threshold:
    def __init__(self, u_model, opponent_model):
        self._u_model = u_model
        self._opponent_model = opponent_model
    
    def calc(self, relative_time):
        edge_concession = self._opponent_model.calc_concession()
        relative = 1 - relative_time ** edge_concession
        return self._u_model.get_by_relative(relative)

class SubNegotiator:
    def __init__(self, sub_negotiation, all_bids):
        _neg_info, _cntxt = sub_negotiation
        self._u_model = UtilModel(_cntxt['ufun'], all_bids)
        self._opponent_model = OpponentModel(all_bids)
        self._threshold = Threshold(self._u_model, self._opponent_model)
        
        # for step in range(_neg_info.nmi.n_steps):
        #     th = self.threshold.calc(step / _neg_info.nmi.n_steps)
        #     print(f'[{step}]th={th}', end='')
        #     print([self.u_model.get(bid) for bid in all_bids if self.u_model.get(bid) >= th])
        
        # for i, area in enumerate(self.u_model._areas):
        #     print(f'[{area.min_u:.3f}〜{area.max_u:.3f}]<{area.min_ru:.3f}〜{area.max_ru:.3f}>: ', end='')
        #     print([f'{self.u_model.get(b):.3f}' for b in area.all_bids])

        # exit()

        self._curr_step = 0
    
    def update(self, state):
        if self._curr_step == state.step:
            return
        self._curr_step = state.step
    
    def proposal(self, state):
        th = self._threshold.calc(state.relative_time)
        target_bids = [bid for bid in self._u_model.all_bids if self._u_model.get(bid) >= th]
        return random.choice(target_bids)

    def respond(self, state):
        my_util = self._u_model.get(state.current_offer)
        th = self._threshold.calc(state.relative_time)
        if my_util >= th:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

# TODO 次に Accept/Reject/EndNeg 戦略を考える 
# TODO その次に MCTS-Random-Time エージェントを作る（←時間なかったら諦める）
class RifalAgent(ANL2025Negotiator):
    def init(self):
        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        # self.id_dict = {}
        # for neg_id in self.negotiators.keys():
        #     index = self.negotiators[neg_id].context['index']
        #     self.id_dict[index] = neg_id

        self.sub_negs = []
    
    @property
    def curr_sub_neg_idx(self):
        return len(self.sub_negs) - 1

    @property
    def curr_sub_neg(self):
        return self.sub_negs[self.curr_sub_neg_idx]
    
    def _update(self, negotiator_id: str, state: SAOState):
        if self.curr_sub_neg_idx < len(self.finished_negotiators):
            self.sub_negs.append(
                SubNegotiator(
                    sub_negotiation = self.negotiators[negotiator_id],
                    all_bids = self.ufun.outcome_spaces[self.curr_sub_neg_idx].enumerate_or_sample(), 
                )
            )
        else:
            self.curr_sub_neg.update(state)

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self._update(negotiator_id, state)
        return self.curr_sub_neg.proposal(state)

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self._update(negotiator_id, state)
        return self.curr_sub_neg.respond(state)

# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    #Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    #Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(RifalAgent, small=True)
