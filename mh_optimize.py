import pandas as pd
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, LpBinary, PULP_CBC_CMD

def mh_optimize(df_pred, alpha=1., aggregate='majority', debug=False):
    n = (len(df_pred.columns) - 1) // 2  ##  Number of models
    m = len(df_pred)  ##  Number of samples
    ##  MILP optimization...
    prob = LpProblem("Optimization", LpMinimize)
    ##  Variables...
    w_list = [LpVariable(name=f'w_{j}', lowBound=0, upBound=2) for j in range(n)]
    x_list = [LpVariable(name=f'x_{i}', cat=LpBinary) for i in range(m)]
    if aggregate == 'all':
        z_list = (df_pred[[f'p_l_{i}' for i in range(n)]].nunique(axis=1) == 1).astype(int).tolist()
        b_list = (df_pred[[f'p_l_{i}' for i in range(n)] + ['y']].nunique(axis=1) == 1).astype(int).tolist()
    elif aggregate == 'majority':
        # Define z_list: 1 if majority agree (> n/2), 0 otherwise.
        z_list = df_pred.apply(lambda row: int(row[[f'p_l_{i}' for i in range(n)]].value_counts().max() > n/2), axis=1).tolist()
        b_list = (df_pred[[f'p_l_{i}' for i in range(n)]].mode(axis=1)[0] == df_pred['y']).astype(int).tolist()
    
    ##  Objective...
    manual_effort = lpSum(1 - x_list[i]*z_list[i] for i in range(m))
    prob += manual_effort
    ##  Constraints...
    M = 10
    eps = 1e-6
    for i in range(m):
        prob += (lpSum(w_list[j] * df_pred.iloc[i][f'p_theta_{j}'] for j in range(n)) - 1 - M * x_list[i] <= 0, f'x_{i}_constraint_1')
        prob += (lpSum(w_list[j] * df_pred.iloc[i][f'p_theta_{j}'] for j in range(n)) - 1 + M * (1 - x_list[i]) - eps >= 0, f'x_{i}_constraint_2')
    prob += ((lpSum((b_list[i] - z_list[i]) * x_list[i] + 1 for i in range(m)))/m >= alpha, 'Accuarcy constraint')
    ##  Solve...
    gap_rel = 0.01     # 1% gap
    threads = 4
    presolve = 'on'
    status = prob.solve(PULP_CBC_CMD(msg=debug, presolve=presolve, gapRel=gap_rel, threads=threads))
    w_list = [v.varValue for v in w_list]
    x_list = [v.varValue for v in x_list]
    effort_value = prob.objective.value()
    if debug:
        print(prob)
        print('b_list:', b_list)
        print('z_list:', z_list)
        print(LpStatus[status])
        print(status)
        print('Optimal value:', prob.objective.value())
        print('Optimal solution:')
        for v in prob.variables():
            print(v.name, '=', v.varValue)
    return effort_value, w_list, x_list


if __name__ == '__main__':
    df_pred = pd.DataFrame({
        'p_l_0': [0, 0, 0, 1],
        'p_theta_0': [0.9, 0.8, 0.7, 0.6],
        'p_l_1': [0, 1, 1, 1],
        'p_theta_1': [0.4, 0.8, 0.7, 0.6],
        'p_l_2': [0, 0, 2, 0],
        'p_theta_2': [0.9, 0.8, 0.7, 0.6],
        'y': [0, 1, 1, 1],
    })
    print(df_pred)
    effort, w_list, x_list = mh_optimize(df_pred, 1, 'majority', debug=True)
    print('Effort:', effort)
    print('w_list:', w_list)
    print('x_list:', x_list)