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
    def __init__(self, sni, agreements, neg_index, parent=None):
        self.sni = sni

        self.neg_index = neg_index
        self.parent = parent

        self.agreements = agreements
        if self.parent is not None:
            self.outcome = agreements[-1]
            self.outcome_index = self.sni.outcomes.index(self.outcome)
        
        self.children = []
        if not self.is_leaf:
            for child_outcome in self.sni.outcomes:
                node = OutcomeNode(self.sni, 
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
        return self.sni.neg_num - self.neg_index
    
    @property
    def is_leaf(self):
        return self.rest_neg_num == 0
    
    def get_bids(self):
        if self.is_leaf:
            if self.sni.neg_num > 1:
                bid = self.agreements
            else:
                bid = self.outcome
            return [bid]

        bids = []
        for child in self.children:
            bids += child.get_bids()
        
        return bids

class RivUtilSpace:
    def __init__(self, sni):
        self.sni = sni

        self.bid_tree = OutcomeNode(self.sni, 
            agreements = sni.agreements, 
            neg_index = sni.neg_index
        )
        bids = self.bid_tree.get_bids()
        
        self.bid2u = {}
        for bid in bids:
            if self.sni.neg_num == 1:
                u = self.sni.side_ufun(bid)
            else:
                if self.sni.rest_neg_num == 1:
                    u = self.sni.ufun(bid)
                else:
                    ru = self.sni.ufun(bid)

                    n_have_to_accept = sum([1 for o in bid[self.sni.neg_index+1:] if o is not None], 0)
                    ease = 1.0 - (n_have_to_accept / self.sni.rest_neg_num)

                    u = self.sni.coeff['util_weight'] * ru + self.sni.coeff['ease_weight'] * ease
            self.bid2u[str(bid)] = u

        self.descend_bids = sorted(bids, key=lambda bid: self.bid2u[str(bid)], reverse=True)

        self.max_u = self.get(self.descend_bids[0])
        self.outcome_2_max_u = {}
        for bid in self.descend_bids:
            key = str(self.get_curr_outcome(bid))
            if key not in self.outcome_2_max_u.keys():
                self.outcome_2_max_u[key] = self.get(bid)

        self.max_end_neg_u = self.outcome_2_max_u[str(None)]
        self.have_to_end_neg = self.max_end_neg_u == self.max_u
        
        # print({str(bid):f'{self.get(bid):.3f}' for bid in self.descend_bids})
    
    def get_curr_outcome(self, bid):
        if self.sni.neg_num == 1:
            return bid
        else:
            return bid[self.sni.neg_index]

    def get(self, bid):
        return self.bid2u[str(bid)]

class CurveArea:
    def __init__(self, max_):
        self.max = max_
        self.min = None

        self.r_max = None
        self.r_min = None
    
    @property
    def size(self):
        return self.max - self.min

    @property
    def r_size(self):
        return self.r_max - self.r_min
    
    def calc_by_r(self, r):
        return self.min + self.size * ((r - self.r_min) / self.r_size)

class ThresholdMinCurve:
    def __init__(self, sni, u_space):
        self.max = u_space.max_u
        self.min = u_space.max_end_neg_u
        
        if not u_space.have_to_end_neg:
            self.areas = [CurveArea(max_=self.max)]
            prev_u = self.max
            for bid in u_space.descend_bids[1:]:
                u = u_space.get(bid)
                if u < self.min:
                    break
                if u - prev_u > sni.coeff['th_delta']:
                    self.areas[-1].min = prev_u
                    self.areas.append(CurveArea(max_=u))
                prev_u = u
            self.areas[-1].min = self.min
        
            self.size = sum([area.size for area in self.areas], 0)

            self.areas[0].r_min = 0.0
            self.areas[-1].r_max = 1.0
            for i in range(len(self.areas)-1):
                r = self.areas[i].r_min + (self.areas[i].size / self.size)
                self.areas[i].r_max = r
                self.areas[i+1].r_min = r
            
            # print([{'max':a.max,'min':a.min,'r_max':a.r_max,'r_min':a.r_min} for a in self.areas])

    def calc_by_relative(self, r):
        for i, area in enumerate(self.areas):
            if r >= area.r_min:
                return area.calc_by_r(r)
    
class Threshold:
    def __init__(self, sni, u_space):
        self.sni = sni

        self.lower_curve = ThresholdMinCurve(sni, u_space)
        # exit()

        if not u_space.have_to_end_neg:
            self.r_delta = 1.0 / self.lower_curve.size
    
    def calc_range(self, state):
        r_lower = 1 - state.relative_time ** (self.sni.coeff['th_aggressive'])
        lower = self.lower_curve.calc_by_relative(r_lower)

        r_upper = r_lower + self.r_delta
        if r_upper > 1.0:
            r_upper = 1.0
        upper = self.lower_curve.calc_by_relative(r_upper)

        return {'lower': lower, 'upper': upper}

class SideNegotiatorInfo:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.ufun = main_negotiator.ufun
        self.side_ufun = main_negotiator.negotiators[negid].context['ufun']
        self.outcomes = get_outcome_space_from_index(main_negotiator, neg_index)
        self.agreements = [get_agreement_at_index(main_negotiator,i) 
            for i in range(neg_index)]
        self.neg_index = neg_index
        self.neg_num = get_number_of_subnegotiations(main_negotiator)
        self.rest_neg_num = self.neg_num - self.neg_index
        self.coeff = coeff

class SideNegotiatorStrategy:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.sni = SideNegotiatorInfo(main_negotiator, negid, neg_index, coeff)
        self.u_space = RivUtilSpace(self.sni)
        self.threshold = Threshold(self.sni, self.u_space)
    
    def proposal(self, state):
        if self.u_space.have_to_end_neg:
            return [o for o in self.sni.outcomes if o is not None][0]
        
        th_range = self.threshold.calc_range(state)

        target_outcomes = []
        for bid in self.u_space.descend_bids:
            u = self.u_space.get(bid)
            curr_outcome = self.u_space.get_curr_outcome(bid)
            if u < th_range['lower']:
                continue
            elif u > th_range['upper']:
                break
            
            if curr_outcome not in target_outcomes:
                target_outcomes.append(curr_outcome)
        
        selected_index = np.random.choice(len(target_outcomes))
        return target_outcomes[selected_index]

    def respond(self, state):
        if self.u_space.have_to_end_neg:
            return ResponseType.END_NEGOTIATION
        
        th_range = self.threshold.calc_range(state)
        opponent_u = self.u_space.outcome_2_max_u[str(state.current_offer)]
        if opponent_u >= th_range['lower']:
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.id_dict = {}
        set_id_dict(self)

        self.current_neg_index = -1
        self.side_neg_strategies = []

        self.coeff = {
            'util_weight': 0.8,     # 0.0 <= util_weight <= 1.0
            'ease_weight': 0.2,     # ease_weight = 1.0 - util_weight
            'th_aggressive': 1.5,   # th_aggressive > 0
            'th_delta': 0.1,        # 0.0 < proposal_delta <= 1.0
        }

    @property
    def current_side_neg_strategy(self):
        return self.side_neg_strategies[self.current_neg_index]
    
    def update(self, negotiator_id):
        if did_negotiation_end(self):
            self.side_neg_strategies.append(
                SideNegotiatorStrategy(
                    main_negotiator = self, 
                    negid = negotiator_id, 
                    neg_index = get_current_negotiation_index(self),
                    coeff = self.coeff,
                )
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
