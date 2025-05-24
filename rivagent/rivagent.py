"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Natures
*Authors* Jumpei Kawahara(s250312x@st.go.tuat.ac.jp)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
import itertools
import random
from copy import copy

import numpy as np

from negmas.outcomes import Outcome

from .helpers.helperfunctions import get_current_negotiation_index, get_outcome_space_from_index \
    , get_agreement_at_index, get_nmi_from_index, did_negotiation_end, set_id_dict \
    , get_number_of_subnegotiations

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOController, SAOState
from negmas import (
    DiscreteCartesianOutcomeSpace,
    ExtendedOutcome,
    ResponseType, CategoricalIssue,
)

class UtilArea:
    def __init__(self, min_zu, max_zu, outcomes):
        self.min_zu = min_zu
        self.zu_size = max_zu - min_zu
        self.max_zu = max_zu

        self.outcomes = outcomes
        self.n_outcomes = len(outcomes)
    
        self.min_ru = None
        self.max_ru = None

class UtilSpace:
    def __init__(self, ufun, outcomes, agreements, neg_index, neg_num):
        self.outcomes = outcomes
        self.__init_outcome2zu(ufun, outcomes, agreements, neg_index, neg_num, risk_param=0.0)
        self.__init_areas(outcomes, n_area_splits=5)
    
    def __init_outcome2zu(self, ufun, outcomes, agreements, neg_index, neg_num, risk_param):
        self.outcome2zu = {}
        if neg_index == neg_num - 1:
            for o in outcomes:
                bid = tuple(agreements + [o])
                util = ufun(bid)
                self.outcome2zu[str(o)] = util
        else:
            for o in outcomes:
                next_accept_utils = []
                next_end_neg_util = None
                for no in outcomes:
                    bid = tuple(agreements + [o, no] + [None]*(neg_num-neg_index-2))
                    util = ufun(bid)
                    if no is None:
                        next_end_neg_util = util
                    else:
                        next_accept_utils.append(util)
                next_accept_zu = np.mean(next_accept_utils) - risk_param * np.std(next_accept_utils)
                zu = max(next_accept_zu, next_end_neg_util)
                self.outcome2zu[str(o)] = zu
        self.outcome2zu = {k:v for k,v in \
            sorted(self.outcome2zu.items(), key=lambda x:x[1])}
    
    def __init_areas(self, outcomes, n_area_splits):
        area_size = 1.0 / n_area_splits

        outcome_tmp = copy(outcomes)
        self.areas = []
        for i in range(n_area_splits-1,-1,-1):
            min_zu = area_size * i

            outcomes_in_area = []
            for o in outcome_tmp:
                if self.get_from_outcome(o) >= min_zu:
                    outcomes_in_area.append(o)
            for o in outcomes_in_area:
                outcome_tmp.remove(o)
            
            if len(outcomes_in_area) == 0:
                continue

            if self.n_areas == 0:
                max_zu = None
                for o in outcomes_in_area:
                    u = self.get_from_outcome(o)
                    if (max_zu is None) or u > max_zu:
                        max_zu = u
                max_zu = np.max([self.get_from_outcome(o)
                    for o in outcomes_in_area])
            else:
                max_zu = area_size * (i+1)
            
            area = UtilArea(min_zu, max_zu, outcomes_in_area)
            self.areas.insert(0, area)
        
        total_zu_size = sum([area.zu_size for area in self.areas], 0)
        
        self.areas[0].min_ru = 0.0
        self.areas[self.n_areas-1].max_ru = 1.0
        for i in range(self.n_areas-1):
            r = self.areas[i].min_ru + (self.areas[i].zu_size / total_zu_size)
            self.areas[i].max_ru = r
            self.areas[i+1].min_ru = r
    
    def get_from_outcome(self, outcome):
        return self.outcome2zu[str(outcome)]
    
    def get_outcome_near_threshold(self, threshold):
        for o in self.outcomes:
            if self.get_from_outcome(o) >= threshold:
                return o
        else:
            return self.outcome2zu[self.outcome2zu.keys()[-1]]
    
    def calc_by_ratio(self, relative):
        for i in range(self.n_areas-1,-1,-1):
            min_ru = self.areas[i].min_ru
            if relative >= min_ru:
                r_in_area = (relative - min_ru) / (self.areas[i].max_ru - min_ru)
                return self.areas[i].zu_size * r_in_area + self.areas[i].min_zu
    
    def calc_std(self):
        return np.std(list(self.outcome2zu.values()))

    def calc_max_dist(self):
        all_zus = list(self.outcome2zu.values())
        return np.max(all_zus) - np.min(all_zus)
    
    @property
    def n_areas(self):
        return len(self.areas)

class OpponentModel:
    def __init__(self, util_space, n_steps):
        self.c1 = 1.3
        self.c2 = 1.3
        self.c3 = 0.2

        self.util_space = util_space

        self.max_dist = util_space.calc_max_dist()
        self.__init_difficulty(util_space)
        self.n_smooth_max = int(n_steps * 0.1)

        self.current_my_zu = None
        self.zu_dists = []
    
    def __init_difficulty(self, util_space):
        my_zu_std = util_space.calc_std()
        if my_zu_std < 0.15:
            self.difficulty = self.c2
        elif my_zu_std < 0.3:
            self.difficulty = 1.0
        else:
            self.difficulty = 1 / self.c2
    
    def update_when_proposal(self, my_outcome):
        self.current_my_zu = self.util_space.get_from_outcome(my_outcome)
    
    def update_when_respond(self, opponent_outcome):
        opponent_zu = self.util_space.get_from_outcome(opponent_outcome)
        self.zu_dists.append(abs(opponent_zu - self.current_my_zu))
    
    def calc_n_smooth(self, step):
        return int(min(step * 0.5, self.n_smooth_max))
    
    def calc_alpha(self, step):
        recent_d_std = np.std(self.zu_dists[-self.n_smooth_max:])
        if recent_d_std < 0.05:
            return 0.5 + self.c3
        elif recent_d_std < 0.2:
            return 0.5
        else:
            return 0.5 - self.c3
    
    def calc_concession(self, state):
        if state.relative_time < 0.1:
            return 2.0
        
        start_dist_mean = np.mean(self.zu_dists[-self.calc_n_smooth(state.step):])
        recent_dist_mean = np.mean(self.zu_dists[-self.calc_n_smooth(state.step)*2:])
        alpha = self.calc_alpha(state.step)

        dist_slope = (start_dist_mean - recent_dist_mean) / (start_dist_mean + 1e-6)
        curr_dist_ratio = 1 - recent_dist_mean / self.max_dist
        concession_tmp = alpha * dist_slope + (1-alpha) * curr_dist_ratio

        # return self.c1 ** (concession_tmp*2-1) * self.difficulty
        return 1.0

class Threshold:
    def __init__(self, util_space, opponent_model):
        self.util_space = util_space
        self.opponent_model = opponent_model
    
    def calc(self, state):
        ratio = 1 - state.relative_time ** self.opponent_model.calc_concession(state)
        return self.util_space.calc_by_ratio(ratio)

class SideNegotiatorStrategy:
    def __init__(self, main_negotiator, negid, neg_index):
        self.util_space = UtilSpace(
            ufun = main_negotiator.ufun, 
            outcomes = get_outcome_space_from_index(main_negotiator, neg_index),
            agreements = [get_agreement_at_index(main_negotiator,i) 
                for i in range(neg_index)],
            neg_index = neg_index,
            neg_num = get_number_of_subnegotiations(main_negotiator),
        )
        self.opponent_model = OpponentModel(
            util_space = self.util_space,
            n_steps = get_nmi_from_index(main_negotiator, neg_index).n_steps,
        )
        self.threshold = Threshold(
            util_space = self.util_space, 
            opponent_model = self.opponent_model,
        )
    
    def proposal(self, state):
        th = self.threshold.calc(state)
        target_outcome = self.util_space.get_outcome_near_threshold(th)
        self.opponent_model.update_when_proposal(target_outcome)
        return target_outcome

    def respond(self, state):
        self.opponent_model.update_when_respond(state.current_offer)
        th = self.threshold.calc(state)
        if th <= self.util_space.get_from_outcome(None):
            return ResponseType.END_NEGOTIATION
        
        opponent_zu = self.util_space.get_from_outcome(state.current_offer)
        if opponent_zu >= th:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.id_dict = {}
        set_id_dict(self)

        self.current_neg_index = -1
        self.side_neg_strategies = []

    @property
    def current_side_neg_strategy(self):
        return self.side_neg_strategies[self.current_neg_index]

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        if did_negotiation_end(self):
            self.side_neg_strategies.append(
                SideNegotiatorStrategy(self, negotiator_id, get_current_negotiation_index(self))
            )
        return self.current_side_neg_strategy.proposal(state)

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        return self.current_side_neg_strategy.respond(state)

if __name__ == "__main__":
    from .helpers.runner import run_negotiation, run_for_debug, visualize
    # run_for_debug(RivAgent, small=True)
    results = run_negotiation(RivAgent)
    visualize(results)
