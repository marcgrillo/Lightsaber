import json
import numpy as np

def main():
    try:
        with open('physics_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("physics_results.json not found. Run collect_physics.py first.")
        return

    # Structure: data[regime_id][controller_id] = {'T': ..., 'B': ...}
    
    # We want to find Alpha, Beta such that:
    # Cost(R, C) = Alpha * T + Beta * B
    # R0 -> C0 is min
    # R1 -> C1 is min
    # R2 -> C2 is min

    # Grid Search
    # Alpha: 1e8 to 1e10
    alphas = np.logspace(8, 10, 20)
    # Beta: 1e7 to 1e11 (Lower range to satisfy Ratio < 15)
    betas = np.logspace(7, 11, 20) 
    
    best_weights = None
    max_margin = -1.0
    valid_count = 0
    
    for alpha in alphas:
        for beta in betas:
            valid = True
            current_margin = float('inf')
            
            # Check R0
            cost00 = alpha * data['0']['0']['T'] + beta * data['0']['0']['B']
            cost01 = alpha * data['0']['1']['T'] + beta * data['0']['1']['B']
            cost02 = alpha * data['0']['2']['T'] + beta * data['0']['2']['B']
            
            if cost00 >= cost01 or cost00 >= cost02:
                valid = False
            else:
                current_margin = min(current_margin, (cost01 - cost00)/cost00, (cost02 - cost00)/cost00)

            if not valid: continue

            # Check R1
            cost10 = alpha * data['1']['0']['T'] + beta * data['1']['0']['B']
            cost11 = alpha * data['1']['1']['T'] + beta * data['1']['1']['B']
            cost12 = alpha * data['1']['2']['T'] + beta * data['1']['2']['B']
            
            if cost11 >= cost10 or cost11 >= cost12:
                valid = False
            else:
                current_margin = min(current_margin, (cost10 - cost11)/cost11, (cost12 - cost11)/cost11)
                
            if not valid: continue

            # Check R2
            cost20 = alpha * data['2']['0']['T'] + beta * data['2']['0']['B']
            cost21 = alpha * data['2']['1']['T'] + beta * data['2']['1']['B']
            cost22 = alpha * data['2']['2']['T'] + beta * data['2']['2']['B']
            
            if cost22 >= cost20 or cost22 >= cost21:
                valid = False
            else:
                current_margin = min(current_margin, (cost20 - cost22)/cost22, (cost21 - cost22)/cost22)
                
            if valid:
                valid_count += 1
                if current_margin > max_margin:
                    max_margin = current_margin
                    best_weights = (alpha, beta)
                    
    if best_weights:
        print(f"FOUND VALID WEIGHTS (Count: {valid_count})")
        print(f"Alpha: {best_weights[0]:.2e}")
        print(f"Beta:  {best_weights[1]:.2e}")
        print(f"Margin: {max_margin:.2%}")
        
        # Verify
        a, b = best_weights
        print("\nVerification Table (Relative Cost):")
        print("      C0      C1      C2")
        for r_idx in ['0', '1', '2']:
            costs = []
            for c_idx in ['0', '1', '2']:
                c = a * data[r_idx][c_idx]['T'] + b * data[r_idx][c_idx]['B']
                costs.append(c)
            base = min(costs)
            print(f"R{r_idx}: {costs[0]/base:.4f}  {costs[1]/base:.4f}  {costs[2]/base:.4f}")

    else:
        print("NO VALID WEIGHTS FOUND IN SEARCH GRID.")
        # Print gradients
        print("Gradient info:")
        print(f"R0: T(C1)-T(C0) = {data['0']['1']['T'] - data['0']['0']['T']:.2e}")
        print(f"R0: B(C1)-B(C0) = {data['0']['1']['B'] - data['0']['0']['B']:.2e}")
        print(f"R1: T(C1)-T(C0) = {data['1']['1']['T'] - data['1']['0']['T']:.2e}")
        print(f"R1: B(C1)-B(C0) = {data['1']['1']['B'] - data['1']['0']['B']:.2e}")
        print(f"R2: T(C2)-T(C1) = {data['2']['2']['T'] - data['2']['1']['T']:.2e}")
        print(f"R2: B(C2)-B(C1) = {data['2']['2']['B'] - data['2']['1']['B']:.2e}")

if __name__ == "__main__":
    main()
