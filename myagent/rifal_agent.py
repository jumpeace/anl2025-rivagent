"""
**Submitted to ANAC 2025 Automated Negotiation League**
*Team* type your team name here
*Authors* type your team member names with their emails here

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
# TODO 一旦 Random-Time エージェントを作る
# TODO 次に MCTS-Random-Time エージェントを作る
import itertools
import random

import numpy as np

from negmas.outcomes import Outcome

from .helpers.helperfunctions import set_id_dict, did_negotiation_end, get_target_bid_at_current_index, is_edge_agent, \
    find_best_bid_in_outcomespace, all_possible_bids_with_agreements_fixed, get_outcome_space_from_index
#be careful: When running directly from this file, change the relative import to an absolute import. When submitting, use relative imports.
#from helpers.helperfunctions import set_id_dict, ...

from anl2025.negotiator import ANL2025Negotiator
from negmas.sao.controllers import SAOController, SAOState
from negmas import (
    DiscreteCartesianOutcomeSpace,
    ExtendedOutcome,
    ResponseType, CategoricalIssue,
)


class RifalAgent(ANL2025Negotiator):
    def init(self):
        self.current_neg_index = -1

        # Make a dictionary that maps the index of the negotiation to the negotiator id. The index of the negotiation is the order in which the negotiation happen in sequence.
        self.id_dict = {}
        set_id_dict(self)

        self.n_proposals = len(self.ufun.outcome_spaces)
        print(self.n_proposals)
    
    def _calc_threshold(self, state: SAOState) -> float:
        level = 10 * state.relative_time
        return self._curve.utility_at()
    
    def _get_ufun(self, negotiator_id: str):
        _, cntxt = self.negotiators[negotiator_id]
        return cntxt["ufun"]
    
    def _update(self, negotiator_id: str, state: SAOState):
        if self.current_neg_index != len(self.finished_negotiators):
            self.current_neg_index = len(self.finished_negotiators)
            self.id_dict[self.current_neg_index] = negotiator_id
            ufun = self._get_ufun(negotiator_id)
            bids = self.ufun.outcome_spaces[self.current_neg_index].enumerate_or_sample()
            utils = [ufun(bid) for bid in bids]
            print(utils)
            print(np.mean(utils), np.std(utils), np.max(utils))

    def propose(
            self, negotiator_id: str, state: SAOState, dest: str | None = None
    ) -> Outcome | None:
        self._update(negotiator_id, state)
        # print('proposal', state.step)
        
        ufun = self._get_ufun(negotiator_id)
        outcome_spaces = self.ufun.outcome_spaces[self.current_neg_index].enumerate_or_sample()
        target_bids = [bid for bid in outcome_spaces if ufun(bid) >= 0.1]
        # for bid in target_bids:
        #     print(bid, ufun(bid))
        return random.choice(target_bids)

    def respond(
            self, negotiator_id: str, state: SAOState, source: str | None = None
    ) -> ResponseType:
        # print('respond', state.step)
        ufun = self._get_ufun(negotiator_id)
        my_util = ufun(state.current_offer)
        threshold = 0.9
        if my_util >= threshold:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


# if you want to do a very small test, use the parameter small=True here. Otherwise, you can use the default parameters.
if __name__ == "__main__":
    from .helpers.runner import run_a_tournament
    #Be careful here. When running directly from this file, relative imports give an error, e.g. import .helpers.helpfunctions.
    #Change relative imports (i.e. starting with a .) at the top of the file. However, one should use relative imports when submitting the agent!

    run_a_tournament(RifalAgent, small=True)
