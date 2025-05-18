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
        self.ufun = ufun
        self.all_bids = all_bids
        self.n_bids = len(all_bids)
        self._bid2u = {str(bid):ufun(bid) for bid in self.all_bids}
        self._bid2u = {k:v for k,v in sorted(self._bid2u.items(), key=lambda x:x[1])}

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
    
    def get_bid_near_threshold(self, threshold):
        for bid in self.all_bids:
            if self.get(bid) >= threshold:
                return bid
        else:
            return self._bid2u[self._bid2u.keys()[-1]]
    
    def calc_by_relative(self, relative):
        for i in range(self.n_areas-1,-1,-1):
            min_ru = self._areas[i].min_ru
            if relative >= min_ru:
                r_in_area = (relative - min_ru) / (self._areas[i].max_ru - min_ru)
                return self._areas[i].u_size * r_in_area + self._areas[i].min_u
    
    def calc_std(self):
        return np.std(list(self._bid2u.values()))

    def calc_max_dist(self):
        all_us = list(self._bid2u.values())
        return np.max(all_us) - np.min(all_us)
    
    @property
    def n_areas(self):
        return len(self._areas)
    
    def get_area(self, i):
        return self._areas[i]

class OpponentModel:
    def __init__(self, u_model, n_steps):
        self._c1 = 1.3
        self._c2 = 1.3
        self._c3 = 0.2

        self._max_dist = u_model.calc_max_dist()
        self._set_difficulty(u_model.calc_std())
        self._n_smooth_max = int(n_steps * 0.1)

        self._u_get_fn = u_model.get
        self._curr_my_u = None
        self._u_dists = []
    
    def _set_difficulty(self, my_u_std):
        if my_u_std < 0.15:
            self._difficulty = self._c2
        elif my_u_std < 0.3:
            self._difficulty = 1.0
        else:
            self._difficulty = 1 / self._c2
    
    def update_when_proposal(self, my_bid):
        self._curr_my_u = self._u_get_fn(my_bid)
    
    def update_when_respond(self, opponent_bid):
        opponent_u = self._u_get_fn(opponent_bid)
        self._u_dists.append(abs(opponent_u - self._curr_my_u))
    
    def _calc_n_smooth(self, step):
        return int(min(step * 0.5, self._n_smooth_max))
    
    def _calc_alpha(self, step):
        recent_d_std = np.std(self._u_dists[-self._n_smooth_max:])
        if recent_d_std < 0.05:
            return 0.5 + self._c3
        elif recent_d_std < 0.2:
            return 0.5
        else:
            return 0.5 - self._c3
    
    def calc_concession(self, state):
        if state.relative_time < 0.1:
            return 2.0
        
        start_dist_mean = np.mean(self._u_dists[-self._calc_n_smooth(state.step):])
        recent_dist_mean = np.mean(self._u_dists[-self._calc_n_smooth(state.step)*2:])
        alpha = self._calc_alpha(state.step)

        dist_slope = (start_dist_mean - recent_dist_mean) / (start_dist_mean + 1e-6)
        curr_dist_ratio = 1 - recent_dist_mean / self._max_dist
        concession_tmp = alpha * dist_slope + (1-alpha) * curr_dist_ratio

        return self._c1 ** (concession_tmp*2-1) * self._difficulty

class Threshold:
    def __init__(self, u_model, opponent_model):
        self._u_model = u_model
        self._opponent_model = opponent_model
    
    def calc(self, state):
        relative = 1 - state.relative_time ** self._opponent_model.calc_concession(state)
        print('th::', self._opponent_model.calc_concession(state), relative)
        return self._u_model.calc_by_relative(relative)

class SubNegotiator:
    def __init__(self, sub_negotiation, all_bids):
        _neg_info, _cntxt = sub_negotiation
        self._u_model = UtilModel(_cntxt['ufun'], all_bids)
        self._opponent_model = OpponentModel(self._u_model, _neg_info.nmi.n_steps)
        self._threshold = Threshold(self._u_model, self._opponent_model)
    
    def proposal(self, state):
        th = self._threshold.calc(state)
        target_bid = self._u_model.get_bid_near_threshold(th)
        self._opponent_model.update_when_proposal(target_bid)
        return target_bid

    def respond(self, state):
        self._opponent_model.update_when_respond(state.current_offer)
        my_util = self._u_model.get(state.current_offer)
        th = self._threshold.calc(state)
        if my_util >= th:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

class RivAgent(ANL2025Negotiator):
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

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self._update(negotiator_id, state)
        bid = self.curr_sub_neg.proposal(state)
        th = self.curr_sub_neg._threshold.calc(state)
        print(f'[PROPOSE] neg={len(self.sub_negs)}, step={state.step}, bid={bid}, u={self.curr_sub_neg._u_model.ufun(bid)}, th={th}')
        return bid

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self._update(negotiator_id, state)
        res = self.curr_sub_neg.respond(state)

        if res == ResponseType.ACCEPT_OFFER:
            r = 'ACCEPT'
        elif res ==  ResponseType.REJECT_OFFER:
            r = 'REJECT'
        else:
            r = 'BAD-RESPONSE'
        th = self.curr_sub_neg._threshold.calc(state)
        print(f'[RESPOND] neg={len(self.sub_negs)}, step={state.step}, edge_bid={state.current_offer}, u={self.curr_sub_neg._u_model.ufun(state.current_offer)}, th={th}, res={r}')
        return res

# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    #Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    #Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(RivAgent, small=True)
