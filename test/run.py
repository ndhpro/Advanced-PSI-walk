import sys
import time
from copy import copy, deepcopy
import networkx as nx
import numpy as np
import pandas as pd
from env import Environment
from agent import QLearningTable
from pathlib import Path


def load_graph(path):
    G = nx.MultiDiGraph()
    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if ' ' in line and line.find(' ') == line.rfind(' '):
            (u, v) = line.split(' ')
            G.add_edge(u, v)

    # Add super_root
    child = set()
    for u, v in G.edges():
        if u != v:
            child.add(v)
    start_edge = list()
    for n in G.nodes():
        if len(child) == len(G.nodes) or n not in child:
            start_edge.append(('ndhpro', n))

    G.add_edges_from(start_edge)

    return G


def update(num_episode, env, RL, steps, all_costs):
    for e in range(num_episode):
        i = 0
        cost = 0
        done = False
        state = env.reset()

        while not done:
            avail_action = env.get_avail_action(state)
            action = RL.choose_action(str(state), avail_action)

            state_, reward, done = env.step(action, state)
            table_id = 0 if np.random.uniform() > 0.5 else 1
            cost += RL.learn(str(state), action,
                             reward, str(state_), table_id)

            state = deepcopy(state_)
            # print(state['node'], end=' ')

            i += 1
            if i == 50:
                break
        all_costs.append(cost)
        steps.append(i)
        # print()

        if (e+1) % (num_episode//9) == 0:
            RL.reduce_epsilon()


def get_path(env, Q, j):
    state = env.reset()
    path = list()
    i = 0
    done = 0
    cost = 0
    while not done and i < 50:
        avail_actions = env.get_avail_action(state)
        action = list()
        try:
            state_action = Q.loc[str(state), avail_actions]
        except Exception as e:
            print(e)
            break
        action = state_action.idxmax()
        if i == j:
            action = state_action.replace(state_action.max(), -1).idxmax()
        cost += Q.loc[str(state), action]
        state, _, done = env.step(action, state)
        path.append(str(state['node']))
        if (state['node'], state['node']) in env.graph.edges():
            path.append(str(state['node']))
        i += 1
    return path, cost, i


def get_final_path(path, env, RL, steps, all_costs):
    f_path = Path('test/walk/') / (Path(path).stem + '.txt')
    Q_ = [RL.get_q_table(i) for i in range(2)]
    Q = list()
    qid = 0 if len(Q_[0].index) > len(Q_[1].index) else 1
    for i, row in enumerate(Q_[qid].values):
        try:
            row1 = Q_[1-qid].iloc[i, :].values
            nrow = [row[j] if row[j] > row1[j] else row1[j]
                    for j in range(len(row))]
        except:
            nrow = row
        Q.append(nrow)
    Q = pd.DataFrame(Q, index=Q_[qid].index, columns=Q_[qid].columns)
    all_path = dict()
    path, cost, leng = get_path(env, Q, j=-1)
    max_cost = cost / 2
    all_path[' '.join(path)] = cost
    for i in range(leng-1, -1, -1):
        path, cost, _ = get_path(env, Q, j=i)
        if cost >= max_cost:
            all_path[' '.join(path)] = cost
    with open(f_path, 'w') as f:
        for k, v in all_path.items():
            f.write(k + '\n')
    # RL.plot_results(steps, all_costs)


# Main
def run_file(path):
    print(path)
    G = load_graph(path)
    if len(G.edges()) == 0:
        print('Graph has no edge')
        return
    # print(len(G.nodes()), len(G.edges()), end=' ')
    env = Environment(graph=G, root='ndhpro')
    RL = QLearningTable(actions=G.nodes())
    steps = []
    all_costs = []

    t = time.time()
    n_epoch = 1000
    update(n_epoch, env, RL, steps, all_costs)
    # print(round((time.time()-t), 2))

    get_final_path(path, env, RL, steps, all_costs)


if __name__ == "__main__":
    run_file(sys.argv[1])
