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

class OutcomeNode:
    def __init__(self, config, agreements, neg_index, parent=None):
        self.config = config

        self.neg_index = neg_index
        self.parent = parent

        self.agreements = agreements
        if self.parent is not None:
            self.outcome = agreements[-1]
            self.outcome_index = self.config['outcomes'].index(self.outcome)
        
        self.children = []
        if not self.is_leaf:
            for child_outcome in self.config['outcomes']:
                node = OutcomeNode(self.config, 
                    agreements = self.agreements + [child_outcome],
                    neg_index = self.neg_index + 1, 
                    parent = self,
                )
                self.children.append(node)
    
    @property
    def is_root(self):
        return self.parent is None
    
    @property
    def rest_neg_num(self):
        return (self.config['neg_num'] - 1) - self.neg_index
    
    @property
    def is_leaf(self):
        if self.is_root:
            return False
        if self.rest_neg_num == 0:
            return True
        if self.outcome is None:
            return True
        return False
    
    def calc_optimism(self):
        return self.config['coeff']['optimism_max'] * (1 - (self.neg_index / self.config['neg_num']))
    
    def calc_pu(self):
        if self.is_leaf:
            if self.config['neg_num'] > 1:
                bid = tuple(self.agreements + [None] * self.rest_neg_num)
                return self.config['ufun'](bid)
            else:
                return self.config['side_ufun'](self.outcome)

        child_end_neg_pu = None
        child_accept_pus = []
        for child in self.children:
            child_pu = child.calc_pu()
            if child.outcome is None:
                child_end_neg_pu = child_pu
            else:
                child_accept_pus.append(child_pu)
        
        child_accept_pu = np.mean(child_accept_pus) + self.calc_optimism() * np.std(child_accept_pus)
        pu = max(child_accept_pu, child_end_neg_pu)
        self.do_accept = child_accept_pu >= child_end_neg_pu
        return pu
    
    def calc_sorted_bid_rus(self):
        if self.is_leaf:
            bid = list(self.agreements + [None] * self.rest_neg_num)
            if self.config['neg_num'] > 1:
                ru = self.config['ufun'](tuple(bid))
                return [{'bid': bid, 'ru': ru}]
            else:
                ru = self.config['side_ufun'](bid[0])
                return [{'bid': bid, 'ru': ru}]

        if self.do_accept:
            bid_rus = []
            for child in self.children:
                if child.outcome is not None:
                    bid_rus += child.calc_sorted_bid_rus()
        else:
            for child in self.children:
                if child.outcome is None:
                    bid_rus = child.calc_sorted_bid_rus()
                    break

        return sorted(bid_rus, key=lambda x:x['ru'], reverse=True)

class RivUtilSpace:
    def __init__(self, ufun, side_ufun, outcomes, agreements, neg_index, neg_num, coeff):
        self.config = {
            'ufun': ufun,
            'side_ufun': side_ufun,
            'neg_num': neg_num,
            'outcomes': outcomes, 
            'coeff': coeff,
        }

        self.dummy_head = OutcomeNode(self.config, agreements, neg_index-1)
        self.build_outcome2u()
    
    def build_outcome2u(self):
        self.outcome2u = {}
        for outcome_node in self.dummy_head.children:
            pu = outcome_node.calc_pu()

            sorted_bid_rus = []
            for bid_ru_tmp in outcome_node.calc_sorted_bid_rus():
                bid = sorted(bid_ru_tmp['bid'], key=lambda x:str(x))
                if bid in [x['bid'] for x in sorted_bid_rus]:
                    continue
                sorted_bid_rus.append({'bid': bid, 'ru': bid_ru_tmp['ru']})
            accept_array = [int(outcome is not None) for outcome in sorted_bid_rus[0]['bid']]
            accept_ratio = sum(accept_array, 0) / len(accept_array)
            rest_future_accept_ratio = sum(accept_array[:-1], 0) / (len(accept_array) - 1) \
                if len(accept_array) > 1 else 0

            u = self.config['coeff']['my_weight'] * pu \
                + self.config['coeff']['n_accepts_weight'] * accept_ratio \
                + self.config['coeff']['rest_n_accepts_weight'] * (1.0 - rest_future_accept_ratio)
            self.outcome2u[str(outcome_node.outcome)] = u

        self.sorted_outcomes = sorted(self.config['outcomes'], key=lambda o:self.get_from_outcome(o))
        self.sorted_accept_outcomes = [o for o in self.sorted_outcomes if o is not None]
        self.max_accept_u = self.get_from_outcome(self.sorted_accept_outcomes[-1])
        self.end_neg_u = self.get_from_outcome(None)
        # print({str(o):f'{self.get_from_outcome(o):.3f}' for o in self.sorted_outcomes})
    
    def get_from_outcome(self, outcome):
        return self.outcome2u[str(outcome)]
    
    def get_outcome_near(self, threshold, delta):
        target_outcomes = []
        min_u = None
        for ao in self.sorted_accept_outcomes:
            if len(target_outcomes) == 0:
                u = self.get_from_outcome(ao)
                if u >= threshold:
                    target_outcomes.append(ao)
                    min_u = u
            else:
                if self.get_from_outcome(ao) > min_u + delta:
                    break
                target_outcomes.append(ao)
        
        selected_index = np.random.choice(len(target_outcomes))
        return target_outcomes[selected_index]
    
    def calc_by_ratio(self, ratio):
        return self.max_accept_u * ratio

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
                'optimism_max': 1.0,            # optimism_max >= 0.0
                'my_weight': 0.7,               # 0.0 <= my_weight < 1.0
                'n_accepts_weight': 0.15,        # 0.0 <= n_accepts_weight < 1.0
                'rest_n_accepts_weight': 0.15,        # 0.0 <= rest_n_accepts_weight < 1.0
            },
            'threshold': {
                'aggressive': 1.5,              # aggressive > 0
            },
            'proposal_delta': 0.2,              # 0.0 < proposal_delta <= 1.0
        }

        self.util_space = RivUtilSpace(
            ufun = main_negotiator.ufun, 
            side_ufun = main_negotiator.negotiators[negid].context['ufun'],
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
        ret = self.util_space.get_outcome_near(th, self.coeff['proposal_delta'])
        return ret

    def respond(self, state):
        th = self.threshold.calc(state)
        if th <= self.util_space.end_neg_u:
            return ResponseType.END_NEGOTIATION
        
        opponent_u = self.util_space.get_from_outcome(state.current_offer)
        if opponent_u >= th:
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
    
    def update(self, negotiator_id):
        if did_negotiation_end(self):
            self.side_neg_strategies.append(
                SideNegotiatorStrategy(self, negotiator_id, get_current_negotiation_index(self))
            )

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self.update(negotiator_id)
        return self.current_side_neg_strategy.proposal(state)

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self.update(negotiator_id)
        return self.current_side_neg_strategy.respond(state)

if __name__ == "__main__":
    from .helpers.runner import run_negotiation, run_tournament, run_for_debug, visualize
    # run_for_debug(RivAgent, small=True)
    results = run_negotiation(RivAgent)
    # results = run_tournament(RivAgent)
    # visualize(results)
