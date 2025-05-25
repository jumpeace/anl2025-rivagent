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

class UtilSpace:
    def __init__(self, ufun, outcomes, agreements, neg_index, neg_num, coeff):
        self.outcomes = outcomes
        self.coeff = coeff
        self.__init_outcome2util(ufun, agreements, neg_index, neg_num)
    
    def __init_outcome2util(self, ufun, agreements, neg_index, neg_num):
        def calc_util__n_accepts_by_next(curr_outcome, curr_agreements, curr_neg_index):
            if curr_neg_index == neg_num - 1:
                bid = tuple(curr_agreements + [curr_outcome])
                curr_util = ufun(bid)
                n_accepts_by_next = 0
                return curr_util, n_accepts_by_next
            else:
                next_accept_utils = []
                next_end_neg_util = None
                for next_outcome in self.outcomes:
                    next_util, n_accepts_by_next2 = calc_util__n_accepts_by_next( 
                        curr_outcome = next_outcome, 
                        curr_agreements = curr_agreements + [curr_outcome], 
                        curr_neg_index = curr_neg_index+1, 
                    )
                    if next_outcome is None:
                        next_end_neg_util = next_util
                    else:
                        next_accept_utils.append(next_util)
                curr_risk = self.coeff['base_risk'] * self.coeff['risk_growth'] ** (curr_neg_index - neg_index)
                next_accept_util = np.mean(next_accept_utils) + (self.coeff['optimism'] - curr_risk) * np.std(next_accept_utils)
                curr_util = max(next_accept_util, next_end_neg_util)
                n_accepts_by_next = n_accepts_by_next2 + (1 if next_accept_util > next_end_neg_util else 0)
                return curr_util, n_accepts_by_next

        self.outcome2util = {}
        for outcome in self.outcomes:
            util, n_accepts_by_next = calc_util__n_accepts_by_next( 
                curr_outcome = outcome, 
                curr_agreements = agreements, 
                curr_neg_index = neg_index, 
            )
            n_accepts = n_accepts_by_next + (1 if outcome is not None else 0)
            accept_ratio = n_accepts / (neg_num - neg_index)
            self.outcome2util[str(outcome)] = self.coeff['my_weight'] * util + (1 - self.coeff['my_weight']) * accept_ratio

        self.sorted_outcomes = sorted(self.outcomes, key=lambda o:self.outcome2util[str(o)])
        self.sorted_accept_outcomes = [o for o in self.sorted_outcomes if o is not None]
        self.max_accept_util = self.get_from_outcome(self.sorted_accept_outcomes[-1])
        self.end_neg_util = self.get_from_outcome(None)
    
    def get_from_outcome(self, outcome):
        return self.outcome2util[str(outcome)]
    
    def get_outcome_near(self, threshold, delta):
        target_outcomes = []
        min_util = None
        for ao in self.sorted_accept_outcomes:
            if len(target_outcomes) == 0:
                util = self.get_from_outcome(ao)
                if util >= threshold:
                    target_outcomes.append(ao)
                    min_util = util
            else:
                if self.get_from_outcome(ao) > min_util + delta:
                    break
                target_outcomes.append(ao)
        
        selected_index = np.random.choice(len(target_outcomes))
        return target_outcomes[selected_index]
    
    def calc_by_ratio(self, ratio):
        return self.max_accept_util * ratio

class Threshold:
    def __init__(self, util_space, coeff):
        self.util_space = util_space
        self.coeff = coeff
    
    def calc(self, state):
        ratio = 1 - state.relative_time ** (self.coeff['aggressive'])
        ret = self.util_space.calc_by_ratio(ratio)
        return ret

class SideNegotiatorStrategy:
    def __init__(self, main_negotiator, negid, neg_index):
        self.coeff = {
            'util_space': {
                'optimism': 0.9,
                'base_risk': 0.2,
                'risk_growth': 1.4,
                'my_weight': 1.0,
            },
            'threshold': {
                'aggressive': 2.0,
            },
            'proposal_delta': 0.2,
        }

        self.util_space = UtilSpace(
            ufun = main_negotiator.ufun, 
            outcomes = get_outcome_space_from_index(main_negotiator, neg_index),
            agreements = [get_agreement_at_index(main_negotiator,i) 
                for i in range(neg_index)],
            neg_index = neg_index,
            neg_num = get_number_of_subnegotiations(main_negotiator),
            coeff = self.coeff['util_space'],
        )
        self.threshold = Threshold(
            util_space = self.util_space, 
            coeff = self.coeff['threshold'],
        )
    
    def proposal(self, state):
        th = self.threshold.calc(state)
        return self.util_space.get_outcome_near(th, self.coeff['proposal_delta'])

    def respond(self, state):
        th = self.threshold.calc(state)
        # if th <= self.util_space.end_neg_util:
        #     return ResponseType.END_NEGOTIATION
        
        # opponent_util = self.util_space.get_from_outcome(state.current_offer)
        # if opponent_util >= th:
        #     return ResponseType.ACCEPT_OFFER
        
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
    # visualize(results)
