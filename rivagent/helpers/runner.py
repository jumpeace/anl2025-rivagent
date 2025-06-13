"""
This is the code that is part of Tutorial 1 for the ANL 2025 competition, see URL.

This code is free to use or update given that proper attribution is given to
the authors and the ANAC 2025 ANL competition.
"""
import pathlib

from anl2025.scenario import MultidealScenario
from anl2025 import (
    run_session,
    make_job_hunt_scenario,
    make_target_quantity_scenario,
    load_example_scenario,
)
from anl2025.tournament import anl2025_tournament
from anl2025.ufun import CenterUFun

import numpy as np
import matplotlib.pyplot as plt

def run_negotiation(center_agent, edge_agents, scenario_name, n_steps = 10):
    scenario_dict = {
        'dinners': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/dinners")),
        'target-quantity': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/TargetQuantity_example")),
        'job-hunt': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/job_hunt_target")),
    }

    results = run_session(
        scenario=scenario_dict[scenario_name],
        center_type=center_agent,
        edge_types=edge_agents,  # type: ignore
        nsteps=n_steps,
        #  verbose=verbose,
        #  keep_order=keep_order,
        #  share_ufuns=share_ufuns,
        #  atomic=atomic,
        #  output=output,
        #  dry=dry,
        #  method=DEFAULT_METHOD,
        #  sample_edges=False,
    )

    # print some results
    print(f"Center utility: {results.center_utility}")
    print(f"Edge Utilities: {results.edge_utilities}")
    print(f"Agreement: {results.agreements}")

    # extra: for nicer lay-outing and more results:
    cfun = results.center.ufun

    assert isinstance(cfun, CenterUFun)
    side_ufuns = cfun.side_ufuns()

    for i, (e, m, u) in enumerate(
        zip(results.edges, results.mechanisms, side_ufuns, strict=True)  # type: ignore
    ):
        print(
            f"{i:02}: Mechanism {m.name} between ({m.negotiator_ids}) ended in {m.current_step} ({m.relative_time:4.3}) with {m.agreement}: "
            f"Edge Utility = {e.ufun(m.agreement) if e.ufun else 'unknown'}, "
            f"Side Utility = {u(m.agreement) if u else 'unknown'}"
        )
        for outcome in m.outcome_space.enumerate_or_sample():
            print(f"Outcome: {outcome} SUtility: {u(outcome)}")
    # print(f"Center Utility: {results.center_utility}")

    return results

def run_tournament(my_agent, opponent_agents, scenario_names, n_test):
    scenario_dict = {
        'dinners': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/dinners")),
        'target-quantity': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/TargetQuantity_example")),
        'job-hunt': MultidealScenario.from_folder(pathlib.Path("./official_test_scenarios/job_hunt_target")),
    }
    scenarios = [scenario_dict[name] for name in scenario_names]
    competitors = [my_agent] + opponent_agents

    results_history = []
    while len(results_history) < n_test:
        try:
            results = anl2025_tournament(
                scenarios=scenarios,
                n_jobs=-1,
                competitors=competitors,
                verbose=False,
                #  no_double_scores=False,
            )
        except Exception:
            continue
        results_history.append(results)
    
    import numpy as np
    import pandas as pd

    keys = results_history[0].final_scores.keys()
    final_scores = {k:[] for k in keys}
    weighted_average = {k:[] for k in keys}
    for results in results_history:
        for k in keys:
            final_scores[k].append(results.final_scores[k])
            weighted_average[k].append(results.weighted_average[k])
    
    for k in keys:
        fn1 = lambda value: f'{value:.3f}'
        fn2 = lambda values: {
            'min': fn1(np.min(values)),
            'Q1': fn1(pd.Series(values).quantile(0.25)),
            'mean': fn1(np.mean(values)),
            'Q3': fn1(pd.Series(values).quantile(0.75)),
            'max': fn1(np.max(values)),
            'std': fn1(np.mean(values)),
        }
        print(f'{k}:', {
            'final_scores': fn2(final_scores[k]),
            'weighted_average': fn2(weighted_average[k]),
        })

    return results

def visualize(results):
    for _, m in enumerate(results.mechanisms):
        plot_result(m)

def plot_result(m):
    m.plot(save_fig=False)
    plt.show()
    plt.close()
