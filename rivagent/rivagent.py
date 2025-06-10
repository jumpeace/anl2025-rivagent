"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* Natures
*Authors* Jumpei Kawahara(s250312x@st.go.tuat.ac.jp)

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
from collections import defaultdict, Counter
import itertools
import random
from copy import copy

import numpy as np

from .helpers.helperfunctions import get_current_negotiation_index, get_agreement_at_index \
    , get_number_of_subnegotiations, all_possible_bids_with_agreements_fixed, is_edge_agent \
    , get_nmi_from_index

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

def normalize(ary):
    total = sum(ary)
    if total == 0:
        return [1.0 / len(ary) for x in ary]
    return [x / total for x in ary]

class Config:
    def __init__(self, agent):
        self.id_dict = {}
        for neg_id in agent.negotiators.keys():
            index = agent.negotiators[neg_id].context['index']
            self.id_dict[index] = neg_id
        agent.id_dict = self.id_dict

        self.coeff = {
            'pref_gamma': 0.8,          # 0.0 <= pref_gamma <= 1.0
            'oap_init': 0.5,            # 0.0 <= opa_init <= 1.0
            'oap_rt_min1': 0.5,         # 0.0 <= opa_needed_rt_min <= 1.0
            'oap_rt_min2_n_offer': 2.0, # opa_needed_rt_min >= 1.0
            'oap_min': 0.2,             # 0.0 <= opa_min <= 1.0
            'oap_max': 0.5,             # 0.0 <= opa_max <= 1.0
            'oap_gamma': 0.7,           # 0.0 <= oap_gamma <= 1.0
            'th_min_ratio': 0.5,        # 0.0 <= th_min_ratio <= 1.0
            'th_aggressive': 1.0,       # th_aggressive > 0.0
            'th_delta_r': 0.1,          # 0.0 < proposal_delta <= 1.0
        }

        self.neg_num = get_number_of_subnegotiations(agent)

        self.is_edge = is_edge_agent(agent)

        self.first_neg_index = self.collect_first_neg_index(agent)

        # self.n_steps = get_nmi_from_index(agent, self.first_neg_index).n_steps

        self.first_side_ufun = self.collect_first_side_ufun(agent)
        self.center_ufun = lambda bid: agent.ufun.eval_with_expected(bid, use_expected=False)

        self.is_multi_agree = self.collect_is_multi_agree(agent)
        self.use_single_neg = self.is_edge or self.is_multi_agree

        # self.all_offers = self.collect_all_offers(agent)
        # self.n_offers = len(self.all_offers)
        # self.all_outcomes = self.all_offers + [None]
        # self.n_outcomes = len(self.all_outcomes)
        # self.n_issues = self.collect_n_issues(agent)
        # self.issue_n_values_dict = self.collect_issue_n_values_dict(agent)
    
        self.ufun = self.first_side_ufun if self.use_single_neg else self.center_ufun

    def collect_first_neg_index(self, agent):
        return list(self.id_dict.keys())[0] if self.is_edge else 0

    def collect_first_side_ufun(self, agent):
        first_neg_id = self.id_dict[self.first_neg_index]
        return agent.negotiators[first_neg_id].context['ufun']

    def collect_is_multi_agree(self, agent):
        if self.is_edge:
            return False
        
        sample_size, sample_count = 10, 0
        all_bids_tmp = all_possible_bids_with_agreements_fixed(agent)
        all_accept_bids_tmp = [bid for bid in all_bids_tmp
            if sum([int(o is not None) for o in bid]) == self.neg_num]
        for bid in all_accept_bids_tmp:
            if sum([int(o is not None) for o in bid]) < self.neg_num:
                continue
            side_us = [self.first_side_ufun(o) for i, o in enumerate(bid)]
            center_u = self.center_ufun(bid)
            
            if np.max(side_us) != center_u:
                return False
            
            sample_count += 1
            if sample_count >= sample_size:
                return True
    
    # def collect_all_offers(self, agent):
    #     if self.is_edge:
    #         return agent.ufun.outcome_space.enumerate_or_sample()
    #     else:
    #         return agent.ufun.outcome_spaces[0].enumerate_or_sample()
    
    # def collect_n_issues(self, agent):
    #     return len(self.all_offers[0])
    
    # def collect_issue_n_values_dict(self, agent):
    #     issue_values_dict = {i: set() for i in range(self.n_issues)}
    #     for offer in self.all_offers:
    #         for i, value in enumerate(offer):
    #             issue_values_dict[i].add(value)
    #     return {i: len(issue_values_dict[i]) for i in range(self.n_issues)}
    
    # def collect_all_bids(self, agent):
    #     if self.use_single_neg:
    #         return self.all_outcomes
    #     else:
    #         return all_possible_bids_with_agreements_fixed(agent)

class ScoreTree:
    def __init__(self, config, agreements, max_depth, is_root, all_outcomes, oap):
        self._config = config

        self.depth = len(agreements)
        self.max_depth = max_depth
        self.is_root = is_root

        self.agreements = agreements
        if not self.is_root:
            self.outcome = self.agreements[-1]

        if not self.is_leaf:
            self.noc2child = {}
            for noc in all_outcomes:
                child = ScoreTree(config,  
                    agreements = self.agreements + [noc],
                    max_depth = self.max_depth,
                    is_root = False,
                    oap = oap,
                    all_outcomes = all_outcomes,
                )
                self.noc2child[str(noc)] = child

            def judge_append_as_child_fn(noc):
                if noc is None:
                    return True
                return self.get_child(noc).score > self.get_child(None).score
                
            target_nocs = []
            for noc in all_outcomes:
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
            oap = self.calc_decayed_oap_sum(agent),
            all_outcomes = agent.all_outcomes,
        )

        # if not self._config.is_edge:
        #     score_dict = {str(oc): f"{float(self.get(oc)):.3f}" for oc in self.descend_outcomes}
        #     print(f'[{agent.neg_index}] agree: {agent.agreements}(u={self._config.ufun(agent.agreements + [None]*agent.rest_neg_num)}), scores: {score_dict}')
    
    def calc_decayed_oap_sum(self, agent):
        return self._config.coeff['oap_init']

        if len(agent.oap_history) == 0:
            return self._config.coeff['oap_init']
        gamma = self._config.coeff['oap_gamma']
        weighted_sum, weight_sum = 0.0, 0.0
        for i, oap in enumerate(reversed(agent.oap_history)):
            weighted_sum += gamma ** i * oap
            weight_sum += gamma ** i
        ratio = weighted_sum / weight_sum
        return self._config.coeff['oap_min'] + (self._config.coeff['oap_max']-self._config.coeff['oap_min']) * ratio

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
        
        self.size = sum([area.rng.size for area in self.areas])

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
        self.n_steps = agent.n_steps
        self.area_space = ThresholdAreaSpace(agent)
    
    def calc_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def calc_r_mn(self, state):
        relative_time = state.step / (self.n_steps - 1)
        return 1 - relative_time ** (self._config.coeff['th_aggressive'])
    
    def calc(self, state):
        r_mn = self.calc_r_mn(state)
        return self.area_space.get_v(r=r_mn)

    def calc_rng(self, state):
        r_mn = self.calc_r_mn(state)
        mn = self.area_space.get_v(r=r_mn)

        r_mx = r_mn + self.area_space.r_delta
        mx = self.area_space.get_v(r=r_mx)

        return Range(mx=mx, mn=mn)

class OfferModel:
    def __init__(self, agent):
        self._config = agent.config
        self._n_offers = agent.n_offers
        self._n_issues = agent.n_issues
        self._issue_n_values_dict = agent.issue_n_values_dict
        self.my_history = []
        self.opponent_history = []
        self.my_issue_value_counter = defaultdict(Counter)
        self.opponent_issue_value_counter = defaultdict(Counter)
    
    def update(self, offer, timing):
        assert timing in ['proposal', 'respond']
        if offer is None:
            return
        
        if timing == 'proposal':
            history = self.my_history
            issue_value_counter = self.my_issue_value_counter
        elif timing == 'respond':
            history = self.opponent_history
            issue_value_counter = self.opponent_issue_value_counter

        history.append(offer)
        for i, value in enumerate(offer):
            issue_value_counter[i][value] += 1
    
    def calc_oap(self):
        return len(set(self.opponent_history)) / self._n_offers
    
    def calc_preferences(self, state, offers):
        opponent_issue_value_score = {i: defaultdict(lambda: 0) for i in range(self._n_issues)}
        for t, offer in enumerate(reversed(self.opponent_history)):
            base_score = self._config.coeff['pref_gamma'] ** t
            for i, value in enumerate(offer):
                opponent_issue_value_score[i][value] += base_score
        
        non_norm_weight_dict = {}
        for i, value_dict in opponent_issue_value_score.items():
            n_values = self._issue_n_values_dict[i]
            non_norm_weight_dict[i] = 1 - len(value_dict) / n_values
        weight_dict = {k:v for k,v in
            zip(non_norm_weight_dict.keys(), normalize(non_norm_weight_dict.values()))}
        
        def calc_preference(offer):
            score = 0
            for i, value in enumerate(offer):
                score += weight_dict[i] * opponent_issue_value_score[i][value]
            return score
    
        return np.array(
            [calc_preference(offer) for offer in offers
        ], dtype=float)

class RivAgent(ANL2025Negotiator):
    def init(self):
        self.config = Config(self)
        self.neg_index = -1

        self.max_u = 0.0

        self.oap_history = []
    
    def update_neg_if_needed(self):
        if self.neg_index >= len(self.finished_negotiators):
            return 
        
        # finalize negotiation
        if self.neg_index >= 0:
            total_steps = len(self.offer_model.opponent_history)
            if total_steps > self.n_steps * self.config.coeff['oap_rt_min1']\
            or total_steps > self.n_offers * self.config.coeff['oap_rt_min2_n_offer']:
                self.oap_history.append(self.offer_model.calc_oap())
        
        # setup negotiation
        self.neg_index = get_current_negotiation_index(self)
        self.rest_neg_num = self.config.neg_num - self.neg_index

        self.n_steps = get_nmi_from_index(self, self.config.first_neg_index).n_steps \
            if self.config.is_edge else get_nmi_from_index(self, self.neg_index).n_steps

        if not self.config.is_edge:
            self.agreements = [get_agreement_at_index(self, i) 
                for i in range(self.neg_index)]
        self.all_offers = self.ufun.outcome_space.enumerate_or_sample() if self.config.is_edge \
            else self.ufun.outcome_spaces[self.neg_index].enumerate_or_sample()
        self.n_offers = len(self.all_offers)
        self.all_outcomes = self.all_offers + [None]
        self.n_outcomes = len(self.all_outcomes)
        self.n_issues = len(self.all_offers[0])

        issue_values_dict = {i: set() for i in range(self.n_issues)}
        for offer in self.all_offers:
            for i, value in enumerate(offer):
                issue_values_dict[i].add(value)
        self.issue_n_values_dict = {i: len(issue_values_dict[i]) for i in range(self.n_issues)}

        self.all_bids = self.all_outcomes if self.config.use_single_neg \
            else all_possible_bids_with_agreements_fixed(self)
        self.n_bids = len(self.all_bids)

        self.max_u = max(max([self.config.ufun(bid) for bid in self.all_bids]), self.max_u)

        self.score_space = ScoreSpace(self)
        self.threshold = Threshold(self)
        self.offer_model = OfferModel(self)

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
        
        if len(target_offers) == 1:
            target_offer = target_offers[0]
        else:
            preferences = self.offer_model.calc_preferences(state, target_offers)
            if preferences.sum() == 0:
                selected_index = np.random.choice(len(target_offers))
            else:
                selected_index = np.argmax(preferences)
            target_offer = target_offers[selected_index]
        
        self.offer_model.update(target_offer, timing='proposal')
        
        return target_offer

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        self.update_neg_if_needed()

        self.offer_model.update(state.current_offer, timing='respond')

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
                Boulware2025,
                # Linear2025,
                # Conceder2025,
            ],
            scenario_name = 'dinners',
            # scenario_name = 'target-quantity',
            # scenario_name = 'job-hunt',
            n_steps = 10,
        )
    
    else:
        from .helpers.runner import run_tournament
        results = run_tournament(
            my_agent = RivAgent,
            opponent_agents = [
                # Random2025,
                # Boulware2025,
                Linear2025,
                # Conceder2025,
            ],
            scenario_names = [
                'dinners',
                # 'target-quantity',
                # 'job-hunt'
            ],
        )
