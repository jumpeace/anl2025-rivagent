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

from .helpers.helperfunctions import get_current_negotiation_index, get_agreement_at_index \
    , get_number_of_subnegotiations, all_possible_bids_with_agreements_fixed, is_edge_agent

from anl2025.negotiator import ANL2025Negotiator
from negmas import ResponseType
from negmas.outcomes import Outcome
from negmas.sao.controllers import SAOState

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

class Config:
    def __init__(self, agent):
        self.id_dict = {}
        for neg_id in agent.negotiators.keys():
            index = agent.negotiators[neg_id].context['index']
            self.id_dict[index] = neg_id
        agent.id_dict = self.id_dict

        self.coeff = {
            'oap_init': 0.4,        # 0.0 <= opa_init <= 1.0
            'oap_gamma': 0.7,       # 0.0 <= oap_gamma <= 1.0
            'th_min_ratio': 0.5,    # 0.0 <= th_min_ratio <= 1.0
            'th_aggressive': 1.0,   # th_aggressive > 0.0
            'th_delta_r': 0.1,      # 0.0 < proposal_delta <= 1.0
        }

        self.neg_num = get_number_of_subnegotiations(agent)

        self.is_edge = is_edge_agent(agent)
        self.first_side_ufun = self.collect_first_side_ufun(agent)
        self.is_multi_agree = self.collect_is_multi_agree(agent)
        self.use_single_neg = self.is_edge or self.is_multi_agree

        self.all_offers = self.collect_all_offers(agent)
        self.n_offers = len(self.all_offers)
        self.all_outcomes = self.all_offers + [None]
        self.n_outcomes = len(self.all_outcomes)
        self.all_bids = self.collect_all_bids(agent)
        self.n_bids = len(self.all_bids)

        self.ufun = self.first_side_ufun if self.use_single_neg else agent.ufun
        self.max_u = max([self.ufun(bid) for bid in self.all_bids])

    def collect_first_side_ufun(self, agent):
        if self.is_edge:
            first_neg_id = list(self.id_dict.values())[0]
        else:
            first_neg_id = self.id_dict[0]
        return agent.negotiators[first_neg_id].context['ufun']

    def collect_is_multi_agree(self, agent):
        if self.is_edge:
            return False
        
        sample_size, sample_count = 10, 0
        all_bids_tmp = all_possible_bids_with_agreements_fixed(agent)
        all_accept_bids_tmp = [bid for bid in all_bids_tmp
            if sum([int(o is not None) for o in bid],0) == self.neg_num]
        for bid in all_accept_bids_tmp:
            if sum([int(o is not None) for o in bid],0) < self.neg_num:
                continue
            side_us = [self.first_side_ufun(o) for i, o in enumerate(bid)]
            center_u = agent.ufun(bid)
            
            if np.max(side_us) != center_u:
                return False
            
            sample_count += 1
            if sample_count >= sample_size:
                return True
    
    def collect_all_offers(self, agent):
        if self.is_edge:
            return agent.ufun.outcome_space.enumerate_or_sample()
        else:
            return agent.ufun.outcome_spaces[0].enumerate_or_sample()
    
    def collect_all_bids(self, agent):
        if self.use_single_neg:
            return self.all_outcomes
        else:
            return all_possible_bids_with_agreements_fixed(agent)

class ScoreTree:
    def __init__(self, config, agreements, max_depth, is_root, oap):
        self._config = config

        self.depth = len(agreements)
        self.max_depth = max_depth
        self.is_root = is_root

        self.agreements = agreements
        if not self.is_root:
            self.outcome = self.agreements[-1]

        if not self.is_leaf:
            self.noc2child = {}
            for noc in self._config.all_outcomes:
                child = ScoreTree(config,  
                    agreements = self.agreements + [noc],
                    max_depth = self.max_depth,
                    is_root = False,
                    oap = oap,
                )
                self.noc2child[str(noc)] = child

            def judge_append_as_child_fn(noc):
                if noc is None:
                    return True
                return self.get_child(noc).score > self.get_child(None).score
                
            target_nocs = []
            for noc in self._config.all_outcomes:
                if judge_append_as_child_fn(noc):
                    target_nocs.append(noc)
            
            self.descend_nocs = sorted(target_nocs, key=lambda noc: self.get_child(noc).score, reverse=True)

            if not self.is_root:
                self.noc2prop = {}
                rest_prop_total = 1.0
                for i, noc in enumerate(self.descend_nocs):
                    if i + 1 < len(self.descend_nocs):
                        prop = rest_prop_total * oap
                        self.noc2prop[str(noc)] = prop
                        rest_prop_total -= prop
                    else:
                        self.noc2prop[str(noc)] = rest_prop_total
    
    @property
    def is_leaf(self):
        return self.depth == self.max_depth
    
    @property
    def score(self):
        assert not self.is_root
        if self.is_leaf:
            bid = self.outcome if self._config.use_single_neg else self.agreements
            ret = self._config.ufun(bid)
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
    def __init__(self, agent):
        self._config = agent.config

        self.tree = ScoreTree(self._config, 
            agreements = [] if self._config.use_single_neg else agent.agreements,
            max_depth = 1 if self._config.use_single_neg else self._config.neg_num,
            is_root = True,
            oap = self.calc_decayed_oap_sum(agent)
        )

        # self.end_neg_u = self._config.ufun(None if self._config.use_single_neg else agent.agreements + [None]*agent.rest_neg_num) 

        # print('edge' if self._config.is_edge else 'center', agent.neg_index, self.end_neg_u, {str(oc): f'{float(self.get(oc)):.3f}' for oc in self.descend_outcomes})
    
    def calc_decayed_oap_sum(self, agent):
        if len(agent.oap_history) == 0:
            return self._config.coeff['oap_init']
        gamma = self._config.coeff['oap_gamma']
        weighted_sum, weight_sum = 0.0, 0.0
        for i, oap in enumerate(reversed(agent.oap_history)):
            weighted_sum += gamma ** i * oap
            weight_sum += gamma ** i
        return weighted_sum / weight_sum

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
    def descend_offers(self):
        return [o for o in self.tree.descend_nocs
            if o is not None]

class CurveArea:
    def __init__(self, mx):
        self.rng = Range(mx=mx, mn=None)
        self.r_rng = Range(mx=None, mn=None)
    
    def get_v(self, r):
        return self.rng.get_v(r=self.r_rng.get_r(v=r))

class ThresholdSpace:
    def __init__(self, agent):
        config = agent.config

        assert 'score_space' in agent.__dict__.keys()
        score_space = agent.score_space

        self.rng = Range(
            mx = score_space.rng.mx,
            mn = max(score_space.end_neg_score, score_space.rng.mx * config.coeff['th_min_ratio'])
        )
        agent.set_have_to_end_neg(self.rng.mx <= self.rng.mn)
        if agent.have_to_end_neg:
            return
        
        self.delta = score_space.rng.get_v(r=config.coeff['th_delta_r'])

class ThresholdAreaSpace:
    def __init__(self, agent):
        score_space = agent.score_space
        th_space = ThresholdSpace(agent)

        if agent.have_to_end_neg:
            return

        self.area_min_size = 1e-6

        self.areas = [CurveArea(mx=th_space.rng.mx)]
        prev_score = th_space.rng.mx
        for offer in score_space.descend_offers[1:]:
            score = score_space.get(offer)
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
            self.areas[-1].rng.mn = score_space.get(score_space.descend_offers[-1])
        
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
    def __init__(self, agent):
        self._config = agent.config
        self.area_space = ThresholdAreaSpace(agent)
    
    def calc_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_r_mn(self, state):
        return 1 - state.relative_time ** (self._config.coeff['th_aggressive'])
    
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
    def __init__(self, agent):
        self._config = agent.config
        self.offer_history = []
    
    def update_when_proposal(self, step, offer):
        if step == 0:
            self.offer_history.append(offer)
    
    def update_when_respond(self, step, offer):
        if step == 0:
            self.offer_history = []
        self.offer_history.append(offer)
    
    def calc_accept_prop(self):
        if len(self.offer_history) == 0:
            n_unique_offers = 1
        else:
            n_unique_offers = len(set(self.offer_history))
        return n_unique_offers / self._config.n_offers

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.config = Config(self)
        self.neg_index = -1

        self.oap_history = []
    
    def update_neg_if_needed(self):
        if self.neg_index >= len(self.finished_negotiators):
            return 
        
        # finalize negotiation
        if self.neg_index >= 0:
            self.oap_history.append(self.opponent_model.calc_accept_prop())
        
        # setup negotiation
        self.neg_index = get_current_negotiation_index(self)
        self.rest_neg_num = self.config.neg_num - self.neg_index

        if not self.config.is_edge:
            self.agreements = [get_agreement_at_index(self, i) 
                for i in range(self.neg_index)]

        self.score_space = ScoreSpace(self)
        self.threshold = Threshold(self)
        self.opponent_model = OpponentModel(self)

    def set_have_to_end_neg(self, arg):
        self.have_to_end_neg = arg

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self.update_neg_if_needed()

        if self.have_to_end_neg:
            return None
        
        th_rng = self.threshold.calc_rng(state)

        target_offers = []
        for offer in reversed(self.score_space.descend_offers):
            score = self.score_space.get(offer)
            if score < th_rng.mn:
                continue
            elif score > th_rng.mx:
                # 緊急エラー対策
                if len(target_offers) == 0:
                    target_offers.append(offer)
                break
            
            if offer not in target_offers:
                target_offers.append(offer)

        selected_index = np.random.choice(len(target_offers))
        target_offer = target_offers[selected_index]
        
        self.opponent_model.update_when_proposal(step=state.step, offer=target_offer)
        
        return target_offer

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self.update_neg_if_needed()

        self.opponent_model.update_when_respond(step=state.step, offer=state.current_offer)

        if self.have_to_end_neg:
            return ResponseType.END_NEGOTIATION
        
        th = self.threshold.calc(state)
        opponent_score = self.score_space.get(state.current_offer)
        if opponent_score >= th:
            return ResponseType.ACCEPT_OFFER
        
        return ResponseType.REJECT_OFFER

if __name__ == "__main__":
    from anl2025.negotiator import Boulware2025, Random2025, Linear2025, Conceder2025

    do_tournament = True

    if not do_tournament:
        from .helpers.runner import run_negotiation
        results = run_negotiation(
            center_agent = RivAgent,
            edge_agents = [
                # Random2025,
                # Boulware2025,
                Linear2025,
                # Conceder2025,
            ],
            # scenario_name = 'dinners',
            scenario_name = 'target-quantity',
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
                'dinners',
                'target-quantity',
                # 'job-hunt'
            ],
        )
