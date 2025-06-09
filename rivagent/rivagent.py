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
    , get_number_of_subnegotiations, all_possible_bids_with_agreements_fixed, get_negid_from_index

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOController, SAOState
from negmas import (
    DiscreteCartesianOutcomeSpace,
    ExtendedOutcome,
    ResponseType, CategoricalIssue,
)

class Range:
    def __init__(self, mx=None, mn=None):
        self.mx = mx
        self.mn = mn
    
    @property
    def size(self):
        return self.mx - self.mn
    
    def get_v(self, r):
        return self.mn + self.size * r
    
    def get_r(self, v):
        return (v - self.mn) / self.size
    
    def __str__(self):
        return f'[{self.mn}~{self.mx}]'

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
        return self.sni.c.neg_num - self.neg_index
    
    @property
    def is_leaf(self):
        return self.rest_neg_num == 0
    
    def get_bids(self):
        if self.is_leaf:
            if self.sni.c.neg_num > 1:
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
        self.is_multi_agreement = self.sni.c.is_multi_agreement
        self.is_use_bid_as_outcome = self.sni.c.neg_num == 1 or self.is_multi_agreement

        self.rng = Range(mx=self.sni.c.max_u, mn=0.0)

        self.end_neg_side_u = self.sni.side_ufun(None) if not self.sni.c.use_outcome else 0.0


        if self.sni.c.use_outcome:
            bids = self.sni.outcomes
        else:
            bid_tree = OutcomeNode(self.sni, 
                agreements = sni.agreements, 
                neg_index = sni.neg_index
            )
            if self.sni.c.n_bids <= self.sni.coeff['n_sample_bids']:
                bids = bid_tree.get_bids()
            else:
                bids = []
                sample_size = self.sni.coeff['n_sample_bids'] // self.sni.n_outcomes
                for child_node in bid_tree.children:
                    child_bids = child_node.get_bids()
                    indices = np.random.choice(len(child_bids), size=sample_size, replace=False)
                    bids += [child_bids[i] for i in indices]

        def calc_n_have_to_accept(bid, start):
            if self.sni.c.use_outcome:
                return int(bid is not None)
            return sum([int(o is not None) for o in bid[start:]], 0)
        
        self.bid2u = {}
        for bid in bids:
            if self.sni.c.use_outcome:
                if self.sni.c.is_multi_agreement:
                    u = self.sni.c.first_ufun(bid)
                else:
                    u = self.sni.side_ufun(bid)
            else:
                if self.sni.rest_neg_num == 1:
                    u = self.sni.c.ufun(bid)
                else:
                    ru = self.sni.c.ufun(bid)
                    if not self.sni.c.is_sum_agreement:
                        ease = 1.0 - (calc_n_have_to_accept(bid, start=self.sni.neg_index+1) / (self.sni.rest_neg_num-1))
                        u = (1.0 - self.sni.coeff['ease_weight']) * ru + self.sni.coeff['ease_weight'] * ease
                    else:
                        concession = calc_n_have_to_accept(bid, start=self.sni.neg_index) / self.sni.rest_neg_num
                        u = (1.0 - self.sni.coeff['concession_weight']) * ru + self.sni.coeff['concession_weight'] * concession

            self.bid2u[str(bid)] = u

        self.descend_bids = sorted(bids, key=lambda bid: self.bid2u[str(bid)], reverse=True)
        self.descend_accept_bids = [bid for bid in self.descend_bids
            if self.get_curr_outcome(bid) is not None]

        self.mx_bid_u = self.get(self.descend_accept_bids[0])
        self.outcome_2_max_u = {}
        for bid in self.descend_accept_bids:
            key = str(self.get_curr_outcome(bid))
            if key not in self.outcome_2_max_u.keys():
                self.outcome_2_max_u[key] = self.get(bid)
    
    def get_curr_outcome(self, bid):
        if self.sni.c.use_outcome:
            return bid
        else:
            return bid[self.sni.neg_index]

    def get(self, bid):
        return self.bid2u[str(bid)]
    
    def get_max_by_outcome(self, outcome):
        return self.outcome_2_max_u[str(outcome)]

class CurveArea:
    def __init__(self, mx):
        self.rng = Range(mx=mx, mn=None)
        self.r_rng = Range(mx=None, mn=None)
    
    def get_v(self, r):
        return self.rng.get_v(r=self.r_rng.get_r(v=r))

class ThresholdSpace:
    def __init__(self, sni, u_space):
        self.rng = Range(
            mx = min(u_space.mx_bid_u, u_space.rng.mx),
            mn = max(u_space.end_neg_side_u, u_space.rng.mx * 0.6)
        )
        sni.set_have_to_end_neg(self.rng.mx <= self.rng.mn)
        
        self.delta = u_space.rng.get_v(r=sni.coeff['th_delta_r'])

class ThresholdAreaSpace:
    def __init__(self, sni, u_space, th_space):
        if sni.have_to_end_neg:
            return

        self.area_min_size = 1e-6

        self.areas = [CurveArea(mx=th_space.rng.mx)]
        prev_u = th_space.rng.mx
        for bid in u_space.descend_accept_bids[1:]:
            u = u_space.get(bid)
            if u < th_space.rng.mn:
                area_mn = th_space.rng.mn
                if prev_u - area_mn > th_space.delta:
                    area_mn = prev_u - th_space.delta
                self.areas[-1].rng.mn = area_mn
                break

            if prev_u - u > th_space.delta:
                if self.areas[-1].rng.mx == prev_u:
                    self.areas[-1].rng.mn = self.areas[-1].rng.mx - self.area_min_size
                else:
                    self.areas[-1].rng.mn = prev_u
                self.areas.append(CurveArea(mx=u))
            prev_u = u
        else:
            self.areas[-1].rng.mn = u_space.get(u_space.descend_accept_bids[-1])
        
        self.size = sum([area.rng.size for area in self.areas], 0)

        self.areas[0].r_rng.mx = 1.0
        self.areas[-1].r_rng.mn = 0.0
        for i in range(len(self.areas)-1):
            r = self.areas[i].r_rng.mx - (self.areas[i].rng.size / self.size)
            self.areas[i].r_rng.mn = r
            self.areas[i+1].r_rng.mx = r

        self.r_delta = (th_space.delta / self.size) if self.size > 0.0 else 0.0

    def get_v(self, r):
        for area in self.areas:
            if r >= area.r_rng.mn:
                return area.get_v(r)
    
class Threshold:
    def __init__(self, sni, u_space):
        self.sni = sni
        self.space = ThresholdSpace(sni, u_space)
        self.area_space = ThresholdAreaSpace(sni, u_space, self.space)
    
    def calc_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_r_mn(self, state):
        return 1 - state.relative_time ** (self.sni.coeff['th_aggressive'])
    
    def calc(self, state):
        r_mn = self.calc_r_mn(state)
        return self.area_space.get_v(r=r_mn)

    def calc_rng(self, state):
        r_mn = self.calc_r_mn(state)
        mn = self.area_space.get_v(r=r_mn)

        r_mx = r_mn + self.area_space.r_delta
        mx = self.area_space.get_v(r=r_mx)

        return Range(mx=mx, mn=mn)

class SideNegotiatorInfo:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.c = main_negotiator.cni

        self.side_ufun = main_negotiator.negotiators[negid].context['ufun']

        if not self.c.is_edge:
            self.outcomes = get_outcome_space_from_index(main_negotiator, neg_index)
        else:
            self.outcomes = main_negotiator.ufun.outcome_space.enumerate_or_sample()
            self.outcomes.append(None)
        self.n_outcomes = len(self.outcomes)

        self.agreements = [get_agreement_at_index(main_negotiator,i) 
            for i in range(neg_index)]

        self.neg_index = neg_index
        self.rest_neg_num = self.c.neg_num - self.neg_index

        self.coeff = coeff
    
    def set_have_to_end_neg(self, v):
        self.have_to_end_neg = v

class SideNegotiatorStrategy:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.sni = SideNegotiatorInfo(main_negotiator, negid, neg_index, coeff)
        self.u_space = RivUtilSpace(self.sni)
        self.threshold = Threshold(self.sni, self.u_space)
    
    def proposal(self, state):
        if self.sni.have_to_end_neg:
            return [o for o in self.sni.outcomes if o is not None][0]
        
        th_rng = self.threshold.calc_rng(state)

        target_outcomes = []
        for bid in reversed(self.u_space.descend_bids):
            u = self.u_space.get(bid)
            curr_outcome = self.u_space.get_curr_outcome(bid)
            if u < th_rng.mn:
                continue
            elif u > th_rng.mx:
                # 緊急エラー対策
                if len(target_outcomes) == 0:
                    target_outcomes.append(curr_outcome)
                break
            
            if curr_outcome not in target_outcomes:
                target_outcomes.append(curr_outcome)

        selected_index = np.random.choice(len(target_outcomes))
        return target_outcomes[selected_index]

    def respond(self, state):
        if self.sni.have_to_end_neg:
            return ResponseType.END_NEGOTIATION
        
        th = self.threshold.calc(state)
        opponent_u = self.u_space.get_max_by_outcome(state.current_offer)
        if opponent_u >= th:
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER

class CenterNegotiationInfo:
    def __init__(self, main_negotiator):
        self.neg_num = get_number_of_subnegotiations(main_negotiator)
        self.ufun = main_negotiator.ufun

        self.all_bids = all_possible_bids_with_agreements_fixed(main_negotiator)
        self.n_bids = len(self.all_bids)

        self.neg_num = get_number_of_subnegotiations(main_negotiator)

        self.init_is_edge()
        self.init_first_ufun(main_negotiator)
        self.init_is_multi_agreement()
        self.use_outcome = self.is_edge or self.is_multi_agreement
        self.init_is_sum_agreement(main_negotiator)
        self.init_max_u()
    
    def init_is_edge(self):
        self.is_edge = self.neg_num == 1
    
    def init_first_ufun(self, main_negotiator):
        if not self.is_edge:
            first_negid = get_negid_from_index(main_negotiator, 0)
        else:
            first_negid = list(main_negotiator.id_dict.values())[0]
        self.first_ufun = main_negotiator.negotiators[first_negid].context['ufun']

    def init_is_multi_agreement(self):
        self.is_multi_agreement = False
        if self.is_edge:
            return
        
        sample_size, sample_count = 10, 0
        all_accept_bids = [bid for bid in self.all_bids
            if sum([int(o is not None) for o in bid],0) == self.neg_num]
        for bid in all_accept_bids:
            if sum([int(o is not None) for o in bid],0) < self.neg_num:
                continue
            side_us = [self.first_ufun(o) for i, o in enumerate(bid)]
            center_u = self.ufun(bid)
            self.is_multi_agreement = np.max(side_us) == center_u
            if not self.is_multi_agreement:
                break
            sample_count += 1
            if sample_count >= sample_size:
                break
    
    def init_is_sum_agreement(self, main_negotiator):
        self.is_sum_agreement = False
        if self.is_edge:
            return

        outcomes = get_outcome_space_from_index(main_negotiator, 0)
        accept_outcomes = [o for o in outcomes if o is not None]
        worst_outcome = sorted(accept_outcomes, key=lambda o: self.first_ufun(o))[0]

        sample_size, sample_count = 5, 0
        for i in range(2**(self.neg_num+1)-1):
            bid = []
            for j in range(self.neg_num):
                bid.append(worst_outcome if i < 2**j else None)
            side_us = [self.first_ufun(o) for i, o in enumerate(bid)]
            center_u = self.ufun(bid)
            # 計算誤差を考慮
            self.is_sum_agreement = center_u-1e-4 < sum(side_us, 0.0) < center_u+1e-4
            if not self.is_sum_agreement:
                break
            sample_count += 1
            if sample_count >= sample_size:
                break

    def init_max_u(self):
        self.max_u = 0.0
        for bid in self.all_bids:
            u = self.ufun(bid) if not self.is_edge else self.first_ufun(bid)
            self.max_u = max(u, self.max_u)

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.id_dict = {}
        set_id_dict(self)

        self.current_neg_index = -1
        self.side_neg_strategy = None

        self.coeff = {
            'ease_weight': 0.2,         # 0.0 <= ease_weight <= 1.0
            'concession_weight': 0.1,   # 0.0 <= concession_weight <= 1.0
            'th_aggressive': 1.5,       # th_aggressive > 0.0
            'th_delta_r': 0.1,          # 0.0 < proposal_delta <= 1.0
            'n_sample_bids': 3000       # n_sample_bids > 0
        }

        self.cni = CenterNegotiationInfo(self)
    
    def update(self, negotiator_id):
        if did_negotiation_end(self):
            self.side_neg_strategy = \
                SideNegotiatorStrategy(
                    main_negotiator = self,
                    negid = negotiator_id, 
                    neg_index = get_current_negotiation_index(self),
                    coeff = self.coeff,
                )

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self.update(negotiator_id)
        return self.side_neg_strategy.proposal(state)

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self.update(negotiator_id)
        return self.side_neg_strategy.respond(state)

if __name__ == "__main__":
    from .helpers.runner import run_negotiation, run_tournament, run_for_debug, visualize
    # run_for_debug(RivAgent, small=True)
    # results = run_negotiation(RivAgent)
    results = run_tournament(RivAgent)
    # visualize(results)
