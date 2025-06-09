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

class ScoreTree:
    def __init__(self, bi, agreements, max_depth, is_root, opponent_accept_prop):
        self.bi = bi

        self.agreements = agreements

        self.depth = len(agreements)
        self.max_depth = max_depth
        self.is_root = is_root

        if not self.is_leaf:
            self.noc2child = {}
            for noc in self.bi.outcomes:
                child = ScoreTree(bi,  
                    agreements = self.agreements + [noc],
                    max_depth = self.max_depth,
                    is_root = False,
                    opponent_accept_prop = opponent_accept_prop,
                )
                self.noc2child[str(noc)] = child

            def judge_append_as_child_fn(noc):
                if noc is None:
                    return True
                return self.get_child(noc).score > self.get_child(None).score
                
            target_nocs = []
            for noc in self.bi.outcomes:
                if judge_append_as_child_fn(noc):
                    target_nocs.append(noc)
            
            self.descend_nocs = sorted(target_nocs, key=lambda noc: self.get_child(noc).score, reverse=True)
            # print(self.agreements, self.descend_nocs)

            if not self.is_root:
                self.noc2prop = {}
                rest_prop_total = 1.0
                for i, noc in enumerate(self.descend_nocs):
                    if i + 1 < len(self.descend_nocs):
                        prop = rest_prop_total * opponent_accept_prop
                        self.noc2prop[str(noc)] = prop
                        rest_prop_total -= prop
                    else:
                        self.noc2prop[str(noc)] = rest_prop_total
    
    @property
    def is_leaf(self):
        return self.depth == self.max_depth
    
    @property
    def score(self):
        if self.is_leaf:
            return self.bi.calc_u(bid=self.agreements)
        else:
            ret = 0.0
            for noc in self.descend_nocs:
                ret += self.noc2prop[str(noc)] * self.get_child(noc).score
            return ret
    
    @property
    def max_child_score(self):
        assert not self.is_leaf
        noc = self.descend_nocs[0]
        return self.get_child(noc).score

    @property
    def end_neg_child_score(self):
        assert not self.is_leaf
        noc = self.descend_nocs[-1]
        return self.get_child(noc).score
    
    def get_child(self, noc):
        return self.noc2child[str(noc)]

class ScoreSpace:
    def __init__(self, sni):
        self.tree = ScoreTree(sni.b, 
            agreements = sni.agreements if not sni.c.use_outcome else [],
            max_depth = sni.c.neg_num if not sni.c.use_outcome else 1,
            is_root = True,
            opponent_accept_prop = sni.c.calc_opponent_accept_prop(),
        )
        # print({str(oc): self.get(oc) for oc in self.descend_outcomes})

    @property
    def rng(self):
        return Range(mx=self.tree.max_child_score, mn=0.0)
    
    @property
    def end_neg_score(self):
        return self.tree.end_neg_child_score
    
    @property
    def descend_outcomes(self):
        return self.tree.descend_nocs

    def get(self, oc):
        return self.tree.get_child(oc).score

    @property
    def descend_accept_outcomes(self):
        return [o for o in self.tree.descend_nocs
            if o is not None]

class CurveArea:
    def __init__(self, mx):
        self.rng = Range(mx=mx, mn=None)
        self.r_rng = Range(mx=None, mn=None)
    
    def get_v(self, r):
        return self.rng.get_v(r=self.r_rng.get_r(v=r))

class ThresholdSpace:
    def __init__(self, sni, score_space):
        self.rng = Range(
            mx = score_space.rng.mx,
            mn = max(score_space.end_neg_score, score_space.rng.mx * sni.coeff['th_min_ratio'])
        )
        # print(self.rng.mx, self.rng.mn)
        sni.set_have_to_end_neg(self.rng.mx <= self.rng.mn)
        if sni.have_to_end_neg:
            return
        
        self.delta = score_space.rng.get_v(r=sni.coeff['th_delta_r'])

class ThresholdAreaSpace:
    def __init__(self, sni, score_space):
        th_space = ThresholdSpace(sni, score_space)

        if sni.have_to_end_neg:
            return

        self.area_min_size = 1e-6

        self.areas = [CurveArea(mx=th_space.rng.mx)]
        prev_score = th_space.rng.mx
        for oc in score_space.descend_accept_outcomes[1:]:
            score = score_space.get(oc)
            if score < th_space.rng.mn:
                area_mn = th_space.rng.mn
                if prev_score - area_mn > th_space.delta:
                    area_mn = prev_score - th_space.delta
                self.areas[-1].rng.mn = area_mn
                break

            if prev_score - score > th_space.delta:
                if self.areas[-1].rng.mx == prev_score:
                    self.areas[-1].rng.mn = self.areas[-1].rng.mx - self.area_min_size
                else:
                    self.areas[-1].rng.mn = prev_score
                self.areas.append(CurveArea(mx=score))
            prev_score = score
        else:
            self.areas[-1].rng.mn = score_space.get(score_space.descend_accept_outcomes[-1])
        
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
    def __init__(self, sni, score_space):
        self.sni = sni
        self.area_space = ThresholdAreaSpace(sni, score_space)
    
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

class OpponentModel:
    def __init__(self, sni):
        self.sni = sni
        self.offer_history = []
    
    def update(self, offer):
        self.offer_history.append(offer)
    
    def calc_accept_prop(self):
        return len(set(self.offer_history)) / self.sni.b.n_offers

class BidInfo:
    def __init__(self, main_negotiator, negid, neg_index):
        self._cni = main_negotiator.cni

        self.side_ufun = main_negotiator.negotiators[negid].context['ufun']

        if not self._cni.is_edge:
            self.outcomes = get_outcome_space_from_index(main_negotiator, neg_index)
        else:
            self.outcomes = main_negotiator.ufun.outcome_space.enumerate_or_sample()
            self.outcomes.append(None)
        self.n_outcomes = len(self.outcomes)
        self.n_offers = self.n_outcomes - 1

    def calc_u(self, bid):
        if self._cni.is_edge:
            return self.side_ufun(bid[0])
        elif self._cni.is_multi_agreement:
            return self._cni.first_ufun(bid[0])
        else:
            return self._cni.ufun(bid)

class SideNegotiatorInfo:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.b = BidInfo(main_negotiator, negid, neg_index)
        self.c = main_negotiator.cni

        if not self.c.is_edge:
            self.agreements = [get_agreement_at_index(main_negotiator,i) 
                for i in range(neg_index)]
        else:
            self.agreements = []

        self.neg_index = neg_index
        self.rest_neg_num = self.c.neg_num - self.neg_index

        self.coeff = coeff
    
    def set_have_to_end_neg(self, v):
        self.have_to_end_neg = v

class SideNegotiatorStrategy:
    def __init__(self, main_negotiator, negid, neg_index, coeff):
        self.sni = SideNegotiatorInfo(main_negotiator, negid, neg_index, coeff)
        self.score_space = ScoreSpace(self.sni)
        self.threshold = Threshold(self.sni, self.score_space)
        self.opponent_model = OpponentModel(self.sni)
    
    def proposal(self, state):
        if self.sni.have_to_end_neg:
            return None
        
        th_rng = self.threshold.calc_rng(state)

        target_outcomes = []
        for oc in reversed(self.score_space.descend_accept_outcomes):
            score = self.score_space.get(oc)
            if score < th_rng.mn:
                continue
            elif score > th_rng.mx:
                # 緊急エラー対策
                if len(target_outcomes) == 0:
                    target_outcomes.append(oc)
                break
            
            if oc not in target_outcomes:
                target_outcomes.append(oc)

        selected_index = np.random.choice(len(target_outcomes))
        return target_outcomes[selected_index]

    def respond(self, state):
        self.opponent_model.update(offer=state.current_offer)

        if self.sni.have_to_end_neg:
            return ResponseType.END_NEGOTIATION
        
        th = self.threshold.calc(state)
        opponent_u = self.score_space.get(state.current_offer)
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

        self.opponent_accept_prop_history = []

        self.init_is_edge()
        self.init_first_ufun(main_negotiator)
        self.init_is_multi_agreement()
        self.use_outcome = self.is_edge or self.is_multi_agreement
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

    def init_max_u(self):
        self.max_u = 0.0
        for bid in self.all_bids:
            u = self.ufun(bid) if not self.is_edge else self.first_ufun(bid)
            self.max_u = max(u, self.max_u)
    
    def update(self, opponent_accept_prop):
        self.opponent_accept_prop_history.append(opponent_accept_prop)
    
    def calc_opponent_accept_prop(self):
        if len(self.opponent_accept_prop_history) == 0:
            return 0.6
        gamma = 0.5
        weighted_sum, weight_sum = 0.0, 0.0
        for i, oap in enumerate(self.opponent_accept_prop_history):
            weighted_sum += gamma ** i * oap
            weight_sum += gamma ** i
        return weighted_sum / weight_sum

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.id_dict = {}
        set_id_dict(self)

        self.current_neg_index = -1
        self.side_neg_strategy = None

        self.coeff = {
            'opponent_accept_prop': 0.5,    # 0.0 <= opponent_accept_prop <= 1.0
            'th_min_ratio': 0.6,            # 0.0 <=th_min_ratio <= 1.0
            'th_aggressive': 1.5,           # th_aggressive > 0.0
            'th_delta_r': 0.1,              # 0.0 < proposal_delta <= 1.0
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
        response = self.side_neg_strategy.respond(state)
        if response != ResponseType.REJECT_OFFER:
            self.cni.update(
                opponent_accept_prop = self.side_neg_strategy.opponent_model.calc_accept_prop()
            )
        return response

if __name__ == "__main__":
    from anl2025.negotiator import Boulware2025, Random2025, Linear2025, Conceder2025

    do_tournament = False

    if not do_tournament:
        from .helpers.runner import run_negotiation
        results = run_negotiation(
            center_agent = RivAgent,
            edge_agents = [
                Random2025,
                Boulware2025,
                Linear2025,
                Conceder2025,
            ],
            scenario_name = 'dinners',
            # scenario_name = 'target-quantity',
            # scenario_name = 'job-hunt',
        )
    
    else:
        from .helpers.runner import run_tournament
        results = run_tournament(
            my_agent = RivAgent,
            opponent_agents = [
                Random2025,
                Boulware2025,
                Linear2025,
                Conceder2025,
            ],
            scenario_names = [
                # 'dinners',
                'target-quantity',
                # 'job-hunt'
            ],
        )
